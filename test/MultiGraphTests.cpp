/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "TestBed.hpp"

namespace RcclUnitTesting
{
    TEST(MultiGraph, Random)
    {
        TestBed testBed;

        // Configuration
        std::vector<std::vector<ncclFunc_t>>     const groupCalls       = {{ncclCollAllReduce, ncclCollAllGather, ncclCollBroadcast}};
                                                                            // {ncclCollAllGather, ncclCollReduceScatter}, 
                                                                            // {ncclCollBroadcast, ncclCollAllToAll}};
        std::vector<ncclRedOp_t>                 const redOps           = {ncclSum, ncclSum, ncclSum};
        std::vector<ncclDataType_t>              const dataTypes        = {ncclFloat16, ncclFloat32, ncclFloat64};
        std::vector<int>                         const numElements      = {1048576, 384 * 1024, 384};
        bool                                     const inPlace          = false;
        bool                                     const useManagedMem    = false;

        bool isCorrect = true;
        for (int groupCallIdx = 0; groupCallIdx < groupCalls.size(); ++groupCallIdx)
        {
            std::vector<ncclFunc_t> funcTypes = groupCalls[groupCallIdx];
            int totalRanks                    = testBed.ev.GetNumGpusList().size();
            int numCollPerGroup               = funcTypes.size();

            testBed.InitComms(TestBed::GetDeviceIdsList(/*single process*/ 1, totalRanks), numCollPerGroup);

            if (testBed.ev.showNames)
                INFO("SP %d-ranks MultiGraph Random\n", totalRanks);
            
            OptionalColArgs options;
            options.redOp = redOps[groupCallIdx];
            options.root = 0;

            for (int collIdx = 0; collIdx < numCollPerGroup; ++collIdx)
            {
                int numInputElements;
                int numOutputElements;
                CollectiveArgs::GetNumElementsForFuncType(funcTypes[collIdx],
                                                        numElements[groupCallIdx],
                                                        totalRanks,
                                                        &numInputElements,
                                                        &numOutputElements);

                testBed.SetCollectiveArgs(funcTypes[collIdx],
                                        dataTypes[groupCallIdx],
                                        numInputElements,
                                        numOutputElements,
                                        options,
                                        collIdx);
            }
            testBed.AllocateMem(inPlace, useManagedMem);
            testBed.PrepareData();
            testBed.ExecuteCollectives({}, true, true);
            // testBed.LaunchGraphs();
            // testBed.ValidateResults(isCorrect);
            // testBed.DeallocateMem();
            // testBed.DestroyComms();
            // testBed.DestroyGraphs();
        }

        for (int i = 0; i < 100; ++i)
        {
            printf("iteration: %d\n", i);
            testBed.LaunchGraphs();
            testBed.ValidateResults(isCorrect);
        }
        testBed.DeallocateMem();
        testBed.DestroyGraphs();
        testBed.DestroyComms();
        testBed.Finalize();
    }
}