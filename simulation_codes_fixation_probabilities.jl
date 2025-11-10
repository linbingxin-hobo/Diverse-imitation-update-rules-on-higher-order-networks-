using Distributed

numLogicCoresUsed = 20
if nprocs() < numLogicCoresUsed
    addprocs(numLogicCoresUsed - nprocs())
    println("Number of Cores:",nprocs())
end

@everywhere begin 
    using Random
    using DelimitedFiles
    using LinearAlgebra
    using Distributions

    struct nodeProperty
        numHyperedgesNode::Int64
        indexHyperedgesNode::Vector{Int64}
    end

    struct hyperedgeProperty
        numNodesInHyperedge::Int64
        indexNodesInHyperedge::Vector{Int64}
    end

    function readFixedHyperNet(networkFolder, networkType)
        fileName = "NetworkStructure_" * networkType * "_Unweighted"
        filePath =  networkFolder * '\\' * fileName
        incidenceMatrixHyperNet = readdlm(filePath * ".txt", ' ', Int)
        NtotalNode = size(incidenceMatrixHyperNet, 1)
        numHyperedges = size(incidenceMatrixHyperNet, 2) 
        networkStructureNodes = Vector{nodeProperty}(undef, NtotalNode)
        networkStructureHyperedges = Vector{hyperedgeProperty}(undef, numHyperedges)
        for iNode in 1:NtotalNode
            numHyperedgesNodeNow = sum(incidenceMatrixHyperNet[iNode,:])
            indexHyperedgesNodeNow = findall(xx -> xx == 1, incidenceMatrixHyperNet[iNode,:])
            networkStructureNodes[iNode] = nodeProperty(numHyperedgesNodeNow, indexHyperedgesNodeNow)
        end
        for iHyperedge in 1:numHyperedges
            numNodesInHyperedgeNow = sum(incidenceMatrixHyperNet[:,iHyperedge])
            indexNodesInHyperedgeNow = findall(xx -> xx == 1, incidenceMatrixHyperNet[:,iHyperedge])
            networkStructureHyperedges[iHyperedge] = hyperedgeProperty(numNodesInHyperedgeNow, indexNodesInHyperedgeNow)
        end
        return networkStructureNodes, networkStructureHyperedges
    end

    function initializeStrategy(Ntotal,initialNumberOfMutants)
        #### randomly select initialNumberOfMutants individual as A-player
        strategyNode = zeros(Int, Ntotal);

        selected_indices = randperm(Ntotal)[1:initialNumberOfMutants]
        for index in selected_indices
            strategyNode[index] = 1
        end
        return selected_indices, 1
    end

    function interactionGames(nodeCalculate, strategyMain, gamePayoff, networkStructureNodes, numAInHyperedgesNow,networkStructureHyperedges,degreeOfNode)

        strategyNode = strategyMain[nodeCalculate]
        payoffNode = 0.0
    
        for hyperedgeNow in networkStructureNodes[nodeCalculate].indexHyperedgesNode
            numACoplayerInHyperedge = numAInHyperedgesNow[hyperedgeNow] - strategyNode
            payoffOneGame = gamePayoff[numACoplayerInHyperedge+1, 2 - strategyNode]
            payoffNode += payoffOneGame
        end

        ### Average payoff
        payoffNode = payoffNode / degreeOfNode
        return payoffNode
    end

    function parameterizedImitatingProcess(nodeChosenUpdate, numAInHyperedgesNow, strategyMain, strategyNodeUpdatePrevious, gamePayoff, networkStructureNodes, networkStructureHyperedges, selectionIntensity,degreeOfNode,sizeOfGroup,s_Hyperedges,q_Neighbors,ifConsiderPersonalInformation)
        strategyNodeChosenUpdate = strategyMain[nodeChosenUpdate]
        HyperedgesOwnedByUpdatedInd = networkStructureNodes[nodeChosenUpdate].indexHyperedgesNode
        IndexCompetitorNeighbors = []
        
        if ifConsiderPersonalInformation == 1
            append!(IndexCompetitorNeighbors, nodeChosenUpdate)
        end

        ### select sq role models 
        selectedHyperedges=sample(HyperedgesOwnedByUpdatedInd, s_Hyperedges, replace=false)
        for iEdge in 1:s_Hyperedges
            indexEdgeNow = selectedHyperedges[iEdge]

            indexOneCompetitor = sample(networkStructureHyperedges[indexEdgeNow].indexNodesInHyperedge, q_Neighbors, replace=false)
            while nodeChosenUpdate in indexOneCompetitor
                indexOneCompetitor = sample(networkStructureHyperedges[indexEdgeNow].indexNodesInHyperedge,q_Neighbors)              
            end 
            append!(IndexCompetitorNeighbors, indexOneCompetitor) 
        end
        ###

        if sum(strategyMain[IndexCompetitorNeighbors]) == 0 
        #### neighbors are all D, the focal individual becomes a D
            strategyMain[nodeChosenUpdate] = 0
            diffStrChangeDirectionNodeUpdate = 0 - strategyNodeChosenUpdate
            return diffStrChangeDirectionNodeUpdate
        elseif sum(strategyMain[IndexCompetitorNeighbors]) == length(IndexCompetitorNeighbors)
        #### neighbors are all C, the focal individual becomes a C
            strategyMain[nodeChosenUpdate] = 1
            diffStrChangeDirectionNodeUpdate = 1 - strategyNodeChosenUpdate
            return diffStrChangeDirectionNodeUpdate
        end


        sumCooperatorFitness = 0.0
        sumTotalNeighFitness = 0.0
        for neighborNow in IndexCompetitorNeighbors
            
            payoffNodeNow = interactionGames(neighborNow, strategyMain, gamePayoff, networkStructureNodes, numAInHyperedgesNow,networkStructureHyperedges,degreeOfNode)
            ##### exponential fitness: f: P -> exp(w * P)
            neighborFitness = exp(selectionIntensity * payoffNodeNow)
            sumTotalNeighFitness += neighborFitness
            sumCooperatorFitness += strategyMain[neighborNow] * 1.0 * neighborFitness
        end

        probToImitateCooperatorNeigh = sumCooperatorFitness / sumTotalNeighFitness

        ##### Neighbors compete for reproduction
        if rand() <= probToImitateCooperatorNeigh
            strategyMain[nodeChosenUpdate] = 1
        else
            strategyMain[nodeChosenUpdate] = 0
        end
        diffStrChangeDirectionNodeUpdate = strategyMain[nodeChosenUpdate] - strategyNodeChosenUpdate
        
        return diffStrChangeDirectionNodeUpdate

    end 

    function oneInvasionTrial(oneStepOneIndividualChangeUpdateRules::F1, numGeneration, networkStructureNodes, networkStructureHyperedges, gamePayoff, selectionIntensity,degreeOfNode,sizeOfGroup,initialNumberOfMutants,s_Hyperedges,q_Neighbors,ifConsiderPersonalInformation) where {F1}

        totalNumNodes = length(networkStructureNodes)
        totalNumHyperedges = length(networkStructureHyperedges)
        strategyNodeUpdatePrevious = zeros(Int, 2)
        numAInHyperedges = zeros(Int, totalNumHyperedges)

        initialAplayers, strChangeDirectionCurrent  = initializeStrategy(totalNumNodes,initialNumberOfMutants)
        for initialAplayer in initialAplayers
            for iHyperedge in networkStructureNodes[initialAplayer].indexHyperedgesNode
                numAInHyperedges[iHyperedge] += 1
            end
        end
        strategyNodeUpdatePrevious[1] = -1
        strategyNodeUpdatePrevious[2] = strChangeDirectionCurrent

        strategyMain = zeros(Int, totalNumNodes)
        for initialAplayer in initialAplayers
            strategyMain[initialAplayer] += strChangeDirectionCurrent
        end
        numAPlayer = initialNumberOfMutants

        for iGeneration in 1 : numGeneration
            nodeChosenUpdate = rand(1:totalNumNodes)     
            strChangeDirectionCurrent = oneStepOneIndividualChangeUpdateRules(nodeChosenUpdate, numAInHyperedges, strategyMain, strategyNodeUpdatePrevious, gamePayoff, networkStructureNodes, networkStructureHyperedges, selectionIntensity,degreeOfNode,sizeOfGroup,s_Hyperedges,q_Neighbors,ifConsiderPersonalInformation)

            if strChangeDirectionCurrent != 0 
                for iHyperedge in networkStructureNodes[nodeChosenUpdate].indexHyperedgesNode
                    numAInHyperedges[iHyperedge] += strChangeDirectionCurrent

                end
            end
            numAPlayer += strChangeDirectionCurrent
            strategyNodeUpdatePrevious[1] = nodeChosenUpdate
            strategyNodeUpdatePrevious[2] = strChangeDirectionCurrent
            if numAPlayer == totalNumNodes
                return [1, 0]
            elseif numAPlayer == 0
                return [0, 1]
            end
        end
        return [0, 0]
    end

end

function calculateFPCorD_MultiCores(oneStepOneIndividualChangeUpdateRules::F1, numRepetition, numGeneration, networkStructureNodes, networkStructureHyperedges, gamePayoff, selectionIntensity,initialNumberOfMutants,s_Hyperedges,q_Neighbors,ifConsiderPersonalInformation) where {F1}

    invasionAndExtinctionTimes = @sync @distributed (+) for iRepetition in 1 : numRepetition
    ####For homogeneous hypergraphs degree & sizeofgroup has only one value
        degreeOfNode = length(networkStructureNodes[1].indexHyperedgesNode)
        sizeOfGroup = length(networkStructureHyperedges[1].indexNodesInHyperedge)
        oneInvasionTrial(oneStepOneIndividualChangeUpdateRules, numGeneration, networkStructureNodes, networkStructureHyperedges, gamePayoff, selectionIntensity,degreeOfNode,sizeOfGroup,initialNumberOfMutants,s_Hyperedges,q_Neighbors,ifConsiderPersonalInformation)
    end
    invasionTimes, extinctionTimes = invasionAndExtinctionTimes
    fpC = (invasionTimes * 1.0) / numRepetition
    return fpC, invasionTimes, extinctionTimes, numRepetition - invasionTimes - extinctionTimes
end

function mainProgramCalculateFpCandD(pairwiseGamePars::Tuple{Float64, Float64, Int, Float64}, invasionTrialsPars::Tuple{Int,Int,String,Float64,Int,Int,Int,Int}, fileIOPars::NTuple{2,String})

    networkFolder, networkType = fileIOPars
    outputFileName01 = networkFolder * "\\" * "ConditionalFixationProb_" * networkType
    numRepetition, numGeneration, gameType,selectionIntensity,initialNumberOfMutants,s_Hyperedges,q_Neighbors,ifConsiderPersonalInformation = invasionTrialsPars
    updateFunctionMain = parameterizedImitatingProcess
    b1Beg, b1Gap, b1Num, cost= pairwiseGamePars

    networkStructureNodesMain, networkStructureHyperedgesMain = readFixedHyperNet(networkFolder, networkType)
    sizeGroup = networkStructureHyperedgesMain[1].numNodesInHyperedge
    degreeOfNode = length(networkStructureNodesMain[1].indexHyperedgesNode)

    #### first column cooperator, second column defector
    gamePayoffMain = zeros(sizeGroup, 2)
    outputFileName = outputFileName01 * "_w" * string(selectionIntensity)*".txt"
    println("outputFileName = $outputFileName")
    open(outputFileName, "a") do io
        writedlm(io, [string("----------------------------------------------------------------------------------------------")])
        writedlm(io, [string("*********** ") string('N') Ntotal string('k') degreeOfNode string('m') sizeGroup string('w') selectionIntensity  string('s') s_Hyperedges string('q') q_Neighbors  strConsiderPersonalInformation "consider personal information" gameType string(" ***********")])
    end

    for rhoCorRhoD in 0:1
        for indB1 in 1 : b1Num
            b1Now = b1Beg + (indB1-1.0) * b1Gap
            println("-------** b1 = $b1Now, w = $selectionIntensity **-------")
            ### gamePayoff = [AA, AB; BA, BB];
    ###############################Different games ######################################
            ##### threshold PGG with threshold = 2 cooperators
            if gameType == "TPGG" 
                for iCoopCoplayer = 0 : sizeGroup - 1
                    if iCoopCoplayer > 0
                        gamePayoffMain[iCoopCoplayer+1, 1] = (iCoopCoplayer+1)*b1Now*1.0/sizeGroup
                    elseif iCoopCoplayer == 0
                        gamePayoffMain[iCoopCoplayer+1, 1] = 0.0
                    end
                    if iCoopCoplayer > 1
                        gamePayoffMain[iCoopCoplayer+1, 2] = iCoopCoplayer*b1Now*1.0/sizeGroup + 1.0
                    elseif iCoopCoplayer <= 1
                        gamePayoffMain[iCoopCoplayer+1, 2] = 0.0
                    end                    
                end
            ##################  MSG ########################
            elseif gameType == "MSG"
                for iCoopCoplayer = 0 : sizeGroup - 1

                    gamePayoffMain[iCoopCoplayer+1, 1] = b1Now-1/(iCoopCoplayer+1)

                    if iCoopCoplayer > 0
                        gamePayoffMain[iCoopCoplayer+1, 2] = b1Now
                    else
                        gamePayoffMain[iCoopCoplayer+1, 2] = 0.0 
                    end                    
                end
            ################## LPGG ########################
            elseif gameType == "LPGG"
                for iCoopCoplayer = 0 : sizeGroup - 1

                    gamePayoffMain[iCoopCoplayer+1, 1] = (iCoopCoplayer+1)*b1Now*1.0/sizeGroup-1
                    gamePayoffMain[iCoopCoplayer+1, 2] = iCoopCoplayer*b1Now*1.0/sizeGroup 
                end
            end
    ##############################################################################
            if rhoCorRhoD == 0
                ### calculate rho_A
                gamePayoff = copy(gamePayoffMain)
               
                fpCorD, invasionTimes, extinctionTimes, coexistenceTimes = calculateFPCorD_MultiCores(updateFunctionMain, numRepetition, numGeneration, networkStructureNodesMain, networkStructureHyperedgesMain, gamePayoff, selectionIntensity,initialNumberOfMutants,s_Hyperedges,q_Neighbors,ifConsiderPersonalInformation)
            elseif rhoCorRhoD == 1
                gamePayoff = reverse(gamePayoffMain) # left and right, up and down
                fpCorD, invasionTimes, extinctionTimes, coexistenceTimes = calculateFPCorD_MultiCores(updateFunctionMain, numRepetition, numGeneration, networkStructureNodesMain, networkStructureHyperedgesMain, gamePayoff, selectionIntensity,initialNumberOfMutants,s_Hyperedges,q_Neighbors,ifConsiderPersonalInformation)
            end
            open(outputFileName, "a") do io 
                if rhoCorRhoD == 0
                    writedlm(io, [string("rNow") b1Now string("fixation probability of C:") fpCorD])
                elseif rhoCorRhoD == 1
                    writedlm(io, [string("rNow") b1Now string("fixation probability of D:") fpCorD])   
                end
            end
        end
    end
    println("Calculation DONE!!")
end

########******************** input parameters *******************

const Ntotal = 500
gameTypeName =  "TPGG" #"MSG"  or "LPGG"  or "TPGG"
selectionIntensityInput = 0.01 # selection intensity $w$
criticalValuePredicted =2.1
initialNumberOfMutantsMain = 1 #initial fraction of mutants $p$=initialNumberOfMutantsMain/Ntotal
considerPersonalInformationWhenUpdatingStrategy = 0 # = 1 (0) means (don't) consider personal inforamtion
s_Hyperedges = 2 ## select s hyperedges randomly
q_Neighbors = 1  ## select q neighbors randomly in each selected hyperedge

########*********************************************************


##################### main
networkType = "HoMo"
numRepetitionInput = 10000000 
numGenerationInput = 3000000
indGraphBeg = 1
indGraph =1
strConsiderPersonalInformation = "don't"
if considerPersonalInformationWhenUpdatingStrategy ==1  
    strConsiderPersonalInformation = ""
end 
println("s=$s_Hyperedges q=$q_Neighbors $strConsiderPersonalInformation consider personal information")
b1GapInput = 0.1 
b1NumInput = 5
costInput = 1.0
networkFolder0 = @__DIR__
networkFolder = networkFolder0 * '\\' * networkType * "-N" * string(Ntotal) * '\\'
println("CriticalValuePredicted = $criticalValuePredicted;")
println("gameType = $gameTypeName")
networkTypeAndIndex = networkType * "_ind" * string(indGraph)
b1BegInput = criticalValuePredicted- b1GapInput * (b1NumInput/2.0-1)   
pairwiseGameParsInput = (b1BegInput, b1GapInput, b1NumInput, costInput)
##### prerun
if indGraph == indGraphBeg
    invasionTrialsParsInput = (10, 20,gameTypeName,1.0,initialNumberOfMutantsMain,s_Hyperedges,q_Neighbors,considerPersonalInformationWhenUpdatingStrategy)
    fileIOParsInput = (networkFolder, networkTypeAndIndex)
    mainProgramCalculateFpCandD(pairwiseGameParsInput, invasionTrialsParsInput, fileIOParsInput)
end
##### running
invasionTrialsParsInput = (numRepetitionInput, numGenerationInput,gameTypeName, selectionIntensityInput,initialNumberOfMutantsMain,s_Hyperedges,q_Neighbors,considerPersonalInformationWhenUpdatingStrategy)
fileIOParsInput = (networkFolder, networkTypeAndIndex)
mainProgramCalculateFpCandD(pairwiseGameParsInput, invasionTrialsParsInput, fileIOParsInput)
