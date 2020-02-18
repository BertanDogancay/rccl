/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "utils.h"
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif
#include "shm.h"

struct p2pConnectInfo {
  int direct;
  union {
    void* directPtr;
    hipIpcMemHandle_t devIpc;
  };
  uint64_t pidHash;
  int id;
  int sendRank;
  int recvRank;
};

struct p2pSendResources {
  struct ncclSendMem* devMem;
  void* ipcPtr;
  uint32_t* next_hdp_reg;  // Next GPU in ring (for p2p transport use only)
  uint64_t* opCount;  // opCount allocated in host memory
  uint64_t* devOpCount;  // device side pointer to opCount
  uint64_t* remOpCount;  // remote opCount allocated in host memory
  uint64_t* devRemOpCount;  // device side pointer to remote opCount
};

struct p2pRecvResources {
  struct ncclRecvMem* devMem;
  void* ipcPtr;
  uint64_t* opCount;  // opCount allocated in host memory
  uint64_t* devOpCount;  // device side pointer to opCount
  uint64_t* remOpCount;  // remote opCount allocated in host memory
  uint64_t* devRemOpCount;  // device side pointer to remote opCount
};

#include <sys/types.h>

NCCL_PARAM(P2pLevel, "P2P_LEVEL", -2);
NCCL_PARAM(P2pDisable, "P2P_DISABLE", -2);

/* Convert a PCI busId string into a local cudaDev device index (cf. CUDA_VISIBLE_DEVICES) */
int busIdToCudaDev(int64_t busId) {
  int ndev;
  if (hipGetDeviceCount(&ndev) != hipSuccess)
    return -1;
  for (int i = 0; i < ndev; i++) {
    char devBusIdStr[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    if (hipDeviceGetPCIBusId(devBusIdStr, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, i) != hipSuccess)
      return -1;
    int64_t devBusId;
    NCCLCHECK(busIdToInt64(devBusIdStr, &devBusId));
    if (busId == devBusId) return i;
  }
  // BusId was not found in our locally visible CUDA devices
  return -1;
}

/* Determine if two peers can communicate through p2p */
ncclResult_t p2pCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  int cpuCount;
  NCCLCHECK(ncclTopoCpuCount(topo, &cpuCount));
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  int p2pLevel = PATH_ARRAY_SIZE;
#else
  // Do not use P2P across sockets by default (provided CUDA permits it).
  // When we are on a single socket, don't even use P2P through the CPU as
  // it should be able to sustain two flows to sysmem faster than PCI P2P.
  int p2pLevel = cpuCount == 1 ? PATH_PHB : PATH_NODE;
#endif
  if (ncclParamP2pDisable() == 1) p2pLevel = 0;
  if (ncclParamP2pLevel() != -2) p2pLevel = ncclParamP2pLevel();

  // Disable P2P
  *ret = 0;

#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  if (!hasFineGrainVramPcie()) return ncclSuccess;
#endif

  if (p2pLevel == 0) return ncclSuccess;

  // Rule out different nodes
  if (info1->hostHash != info2->hostHash) return ncclSuccess;

  // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
  int cudaDev1 = busIdToCudaDev(info1->busId);
  int cudaDev2 = busIdToCudaDev(info2->busId);
  if (cudaDev1 == -1 || cudaDev2 == -1) {
    // Peer's CUDA device is not visible in this process
#if CUDART_VERSION >= 10010
    // But in CUDA 10.1 we can still communicate with 'invisible' devices
    TRACE(NCCL_INIT|NCCL_P2P, "Checking P2P connection between %lx and %lx", info1->busId, info2->busId);
    // Check for NVLink/NVswitch including P2P access
    int nvlink;
    NCCLCHECK(ncclTopoGetNvlink(topo, info1->busId, info2->busId, &nvlink));
    if (nvlink > 0) {
      *ret = 1;
      return ncclSuccess;
    }
#endif
    return ncclSuccess;
  }

  TRACE(NCCL_INIT|NCCL_P2P, "Checking P2P connection between [%d=%lx] and [%d=%lx]", cudaDev1, info1->busId, cudaDev2, info2->busId);

  // Do not detect topology if we're on the same GPU. Note this is not really supported.
  if (cudaDev1 == cudaDev2) {
    *ret = 1;
    return ncclSuccess;
  }

  // See if CUDA can do P2P
  int p2p;
  if (hipDeviceCanAccessPeer(&p2p, cudaDev1, cudaDev2) != hipSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"peer query failed between dev %d(=%lx) and dev %d(=%lx)",
         cudaDev1, info1->busId, cudaDev2, info2->busId);
    return ncclSuccess;
  }
  if (p2p == 0) return ncclSuccess;

  int nvlink = 0;
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
  uint32_t link_type, hops;
  if (hipExtGetLinkTypeAndHopCount(cudaDev1, cudaDev2, &link_type, &hops) != hipSuccess) {
    p2p = 0;
    return ncclSuccess;
  }
  static const char* link_type_name[] = {"HT", "QPI", "PCIE", "IB", "XGMI"};
  static unsigned long long link_status_print_once_mask = 0;
  if (!(link_status_print_once_mask & (1 << (cudaDev1*8 + cudaDev2)))) {
    INFO(NCCL_INIT, "%d -> %d: link type %s hops %d", cudaDev1, cudaDev2,
      link_type_name[link_type], hops);
    link_status_print_once_mask |= (1 << (cudaDev1*8 + cudaDev2));
  }
  if (link_type == HSA_AMD_LINK_INFO_TYPE_XGMI)
    nvlink = (hops == 1);
#else  // Check for NVLink/NVswitch
  NCCLCHECK(ncclTopoGetNvlink(topo, info1->busId, info2->busId, &nvlink));
#endif
  if (nvlink > 0) {
    *ret = 1;
    return ncclSuccess;
  }
  // Finally compute the PCI distance and compare with the p2pLevel.
  int distance;
  NCCLCHECK(ncclTopoGpuDistance(topo, info1->busId, info2->busId, &distance));
  if (distance < p2pLevel) {
    *ret = 1;
  }
  return ncclSuccess;
}

#define TRACE_DUMP_IPC(DEVIPC)                                                             \
  do {                                                                                     \
    unsigned long *devIpc = (unsigned long *) (DEVIPC);                                    \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[0], devIpc[1], devIpc[2], devIpc[3]); \
    TRACE(P2P,"IPC: %016lx %016lx %016lx %016lx", devIpc[4], devIpc[5], devIpc[6], devIpc[7]); \
  } while (0)

#define MAX_SHM_NAME_LEN 1024

/* Send: Create and return connect structures for this peer to connect to me */
ncclResult_t p2pSendSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector* send, int buffSize, int channelId) {
  struct p2pSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;
  int sendSize = sizeof(struct ncclSendMem);
  ALIGN_SIZE(sendSize, CUDA_IPC_MIN);
  NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, sendSize, true));

  resources->next_hdp_reg = 0;
  uint32_t linktype, hops;
  if (hipExtGetLinkTypeAndHopCount(myInfo->cudaDev, peerInfo->cudaDev, &linktype, &hops) != hipSuccess) {
    INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d -> %d failed to get link type and hop count", channelId, myInfo->rank, peerInfo->rank);
    return ncclInternalError;
  }
  if (linktype != HSA_AMD_LINK_INFO_TYPE_XGMI) {
    CUDACHECK(hipDeviceGetAttribute((int*)&resources->next_hdp_reg, hipDeviceAttributeHdpMemFlushCntl,peerInfo->cudaDev));
    TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d -> %d HDP %p", channelId, myInfo->rank, peerInfo->rank, resources->next_hdp_reg);
  }

  struct p2pConnectInfo info;
  info.id = channelId;
  info.pidHash = myInfo->pidHash;
  info.sendRank = myInfo->cudaDev;
  info.recvRank = peerInfo->cudaDev;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-p2p-send-opcount-%lx-%d-%d-%d", info.pidHash, info.id, info.sendRank, info.recvRank);
  TRACE(NCCL_P2P,"Open shmName %s", shmName);
  NCCLCHECK(shmOpen(shmName, sizeof(uint64_t), (void**)&resources->opCount, (void**)&resources->devOpCount, 1));

  if (myInfo->pidHash == peerInfo->pidHash) {
    info.direct = 1;
    info.directPtr = resources->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%d] -> %d[%d] via P2P/common device", channelId, myInfo->rank, myInfo->cudaDev, peerInfo->rank, peerInfo->cudaDev);
      return ncclInternalError;
    } else {
      // Enable P2P access
      hipError_t err = hipDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == hipErrorPeerAccessAlreadyEnabled) {
        hipGetLastError();
      } else if (err != hipSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s",
             peerInfo->cudaDev, peerInfo->busId, err, hipGetErrorString(err));
        return ncclInternalError;
      }
      INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%lx] -> %d[%lx] via P2P/direct pointer",
          channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
    }
  } else {
    // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
    int peerCudaDev = busIdToCudaDev(peerInfo->busId);
    info.direct = 0;
    // Map IPC and enable P2P access
    hipError_t err = hipIpcGetMemHandle(&info.devIpc, (void*)resources->devMem);
    if (err != hipSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d(=%lx) : %d %s",
           myInfo->rank, peerCudaDev, peerInfo->busId, err, hipGetErrorString(err));
      return ncclInternalError;
    }
    INFO(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%lx] -> %d[%lx] via P2P/IPC",
        channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
    //TRACE_DUMP_IPC(&info.devIpc);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Create and return connect structures for this peer to connect to me */
ncclResult_t p2pRecvSetup(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
    struct ncclConnect* connectInfo, struct ncclConnector * recv, int buffSize, int channelId) {

  struct p2pRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;
  int recvSize = offsetof(struct ncclRecvMem, buff)+buffSize;
  ALIGN_SIZE(recvSize, CUDA_IPC_MIN);
  NCCLCHECK(ncclCudaCalloc((char**)&resources->devMem, recvSize, true));

  struct p2pConnectInfo info;
  info.id = channelId;
  info.pidHash = myInfo->pidHash;
  info.sendRank = peerInfo->cudaDev;
  info.recvRank = myInfo->cudaDev;

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-p2p-recv-opcount-%lx-%d-%d-%d", info.pidHash, info.id, info.sendRank, info.recvRank);
  TRACE(NCCL_P2P,"Open shmName %s", shmName);
  NCCLCHECK(shmOpen(shmName, sizeof(uint64_t), (void**)&resources->opCount, (void**)&resources->devOpCount, 1));

  if (myInfo->pidHash == peerInfo->pidHash) {
    info.direct = 1;
    info.directPtr = resources->devMem;
    if (myInfo->cudaDev == peerInfo->cudaDev) {
      TRACE(NCCL_INIT|NCCL_P2P,"%d <- %d via P2P/common device", myInfo->rank, peerInfo->rank);
    } else {
      // Enable P2P access
      hipError_t err = hipDeviceEnablePeerAccess(peerInfo->cudaDev, 0);
      if (err == hipErrorPeerAccessAlreadyEnabled) {
        hipGetLastError();
      } else if (err != hipSuccess) {
        WARN("failed to peer with device %d(=%lx): %d %s",
             peerInfo->cudaDev, peerInfo->busId, err, hipGetErrorString(err));
        return ncclInternalError;
      }
      TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%lx] <- %d[%lx] via P2P/direct pointer", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
    }
  } else {
    // Convert the peer's busId into a local cudaDev index (cf. CUDA_VISIBLE_DEVICES)
    int peerCudaDev = busIdToCudaDev(peerInfo->busId);
    info.direct = 0;
    // Map IPC and enable P2P access
    hipError_t err = hipIpcGetMemHandle(&info.devIpc, (void*)resources->devMem);
    if (err != hipSuccess) {
      WARN("rank %d failed to get CUDA IPC handle to device %d(=%lx) : %d %s",
           myInfo->rank, peerCudaDev, peerInfo->busId, err, hipGetErrorString(err));
      return ncclInternalError;
    }
    TRACE(NCCL_INIT|NCCL_P2P,"Ring %02d : %d[%lx] <- %d[%lx] via P2P/IPC", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
    //TRACE_DUMP_IPC(&info.devIpc);
  }
  static_assert(sizeof(struct p2pConnectInfo) <= sizeof(struct ncclConnect), "p2p Connect Info is too big");
  memcpy(connectInfo, &info, sizeof(struct p2pConnectInfo));
  return ncclSuccess;
}

/* Connect/Send to this peer */
static ncclResult_t p2pSendConnect(struct ncclConnect* connectInfo, struct ncclConnector* send) {
  struct p2pSendResources* resources = (struct p2pSendResources*)send->transportResources;
  struct ncclRecvMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    remDevMem = (struct ncclRecvMem*)(info->directPtr);
    send->conn.direct = 1;
  } else {
    //TRACE_DUMP_IPC(&info->devIpc);
    hipError_t err = hipIpcOpenMemHandle(&resources->ipcPtr, info->devIpc, hipIpcMemLazyEnablePeerAccess);
    remDevMem = (struct ncclRecvMem*)resources->ipcPtr;
    if (err != hipSuccess) {
      WARN("failed to open CUDA IPC handle : %d %s",
          err, hipGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  }

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-p2p-recv-opcount-%lx-%d-%d-%d", info->pidHash, info->id, info->sendRank, info->recvRank);
  TRACE(NCCL_P2P,"Open shmName %s", shmName);
  NCCLCHECK(shmOpen(shmName, sizeof(uint64_t), (void**)&resources->remOpCount, (void**)&resources->devRemOpCount, 0));
  // Remove the file to ensure proper clean-up
  NCCLCHECK(shmUnlink(shmName));

  send->conn.buff = remDevMem->buff;
  send->conn.llBuff = remDevMem->llBuff;
  send->conn.ll128Buff = remDevMem->ll128Buff;
  send->conn.tail = &remDevMem->tail;
  send->conn.opCountRem = resources->devRemOpCount;
  send->conn.head = &resources->devMem->head;
  send->conn.ptrExchange = &resources->devMem->ptrExchange;
  send->conn.opCountLoc = resources->devOpCount;
  send->conn.next_hdp_reg = resources->next_hdp_reg;
  return ncclSuccess;
}

/* Connect/Recv from this peer */
ncclResult_t p2pRecvConnect(struct ncclConnect* connectInfo, struct ncclConnector* recv) {
  struct p2pRecvResources* resources = (struct p2pRecvResources*)recv->transportResources;
  struct ncclSendMem* remDevMem;
  struct p2pConnectInfo* info = (struct p2pConnectInfo*)connectInfo;
  if (info->direct) {
    remDevMem = (struct ncclSendMem*)(info->directPtr);
    recv->conn.direct = 1;
    recv->conn.ptrExchange = &remDevMem->ptrExchange;
  } else {
    //TRACE_DUMP_IPC(&info->devIpc);
    hipError_t err = hipIpcOpenMemHandle(&resources->ipcPtr, info->devIpc, hipIpcMemLazyEnablePeerAccess);
    remDevMem = (struct ncclSendMem*)resources->ipcPtr;
    if (err != hipSuccess) {
      WARN("failed to open CUDA IPC handle : %d %s",
          err, hipGetErrorString(err));
      return ncclUnhandledCudaError;
    }
  }

  char shmName[MAX_SHM_NAME_LEN];
  sprintf(shmName, "nccl-p2p-send-opcount-%lx-%d-%d-%d", info->pidHash, info->id, info->sendRank, info->recvRank);
  TRACE(NCCL_P2P,"Open shmName %s", shmName);
  NCCLCHECK(shmOpen(shmName, sizeof(uint64_t), (void**)&resources->remOpCount, (void**)&resources->devRemOpCount, 0));
  NCCLCHECK(shmUnlink(shmName));

  recv->conn.buff = resources->devMem->buff;
  recv->conn.llBuff = resources->devMem->llBuff;
  recv->conn.ll128Buff = resources->devMem->ll128Buff;
  recv->conn.tail = &resources->devMem->tail;
  recv->conn.opCountLoc = resources->devOpCount;
  recv->conn.head = &remDevMem->head;
  recv->conn.opCountRem = resources->devRemOpCount;
  return ncclSuccess;
}

ncclResult_t p2pSendFree(void* resources) {
  struct p2pSendResources* sendRes = (struct p2pSendResources*)resources;
  if (sendRes->ipcPtr)
    CUDACHECK(hipIpcCloseMemHandle(sendRes->ipcPtr));
  CUDACHECK(hipFree(sendRes->devMem));
  NCCLCHECK(shmClose(sendRes->opCount, sendRes->devOpCount, sizeof(uint64_t)));
  NCCLCHECK(shmClose(sendRes->remOpCount, sendRes->devRemOpCount, sizeof(uint64_t)));
  free(sendRes);
  return ncclSuccess;
}

ncclResult_t p2pRecvFree(void* resources) {
  struct p2pRecvResources* recvRes = (struct p2pRecvResources*)resources;
  if (recvRes->ipcPtr)
    CUDACHECK(hipIpcCloseMemHandle(recvRes->ipcPtr));
  CUDACHECK(hipFree(recvRes->devMem));
  NCCLCHECK(shmClose(recvRes->opCount, recvRes->devOpCount, sizeof(uint64_t)));
  NCCLCHECK(shmClose(recvRes->remOpCount, recvRes->devRemOpCount, sizeof(uint64_t)));
  free(recvRes);
  return ncclSuccess;
}

struct ncclTransport p2pTransport = {
  "P2P",
  p2pCanConnect,
  { p2pSendSetup, p2pSendConnect, p2pSendFree, NULL },
  { p2pRecvSetup, p2pRecvConnect, p2pRecvFree, NULL }
};