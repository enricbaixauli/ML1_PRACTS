# Auxiliary file with functions from previous units
# build class from pract 2

using Statistics

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))

    ann = Chain()
    numInputsLayer = numInputs

    for (i, numOutputsLayer) in enumerate(topology)
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))
        numInputsLayer = numOutputsLayer
    end

    if numOutputs == 1
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ))
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity), softmax)
    end

    return ann
end           

# trainClassANN from pract 3

    
function trainClassANN(topology::AbstractArray{<:Int,1},      
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}} = (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)

    inputs, targets = trainingDataset
    numInputs = size(inputs, 2)
    numOutputs = size(targets, 2)

    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    loss(m, x, y) = (size(y,2) == 1) ? Flux.Losses.binarycrossentropy(m(x), y) : Flux.Losses.crossentropy(m(x), y)
    opt_state = Flux.setup(Flux.Adam(learningRate), ann)

    train_loss_history = Float64[]
    val_loss_history = Float64[]
    test_loss_history = Float64[]

    x_train = inputs'
    y_train = targets'

    x_val, y_val = validationDataset
    x_val = x_val'
    y_val = y_val'

    x_test, y_test = testDataset
    x_test = x_test'
    y_test = y_test'

    # Initial loss (cycle 0)
    push!(train_loss_history, loss(ann, x_train, y_train))
    if size(x_val,2) > 0
        push!(val_loss_history, loss(ann, x_val, y_val))
    end
    if size(x_test,2) > 0
        push!(test_loss_history, loss(ann, x_test, y_test))
    end

    best_val_loss = Inf
    best_ann = deepcopy(ann)
    epochs_since_best = 0

    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(x_train, y_train)], opt_state)
        train_loss = loss(ann, x_train, y_train)
        push!(train_loss_history, train_loss)

        # Validation loss
        val_loss = nothing
        if size(x_val,2) > 0
            val_loss = loss(ann, x_val, y_val)
            push!(val_loss_history, val_loss)
            if val_loss < best_val_loss
                best_val_loss = val_loss
                best_ann = deepcopy(ann)
                epochs_since_best = 0
            else
                epochs_since_best += 1
            end
        end

        # Test loss
        if size(x_test,2) > 0
            test_loss = loss(ann, x_test, y_test)
            push!(test_loss_history, test_loss)
        end

        # Early stopping
        if train_loss <= minLoss
            break
        end
        if size(x_val,2) > 0 && epochs_since_best >= maxEpochsVal
            break
        end
    end

    # Return best ANN if validation set was used, else last ANN
    if size(x_val,2) > 0
        return best_ann, train_loss_history, val_loss_history, test_loss_history
    else
        return ann, train_loss_history, val_loss_history, test_loss_history
    end
end


function trainClassANN(topology::AbstractArray{<:Int,1},  
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; 
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}= 
        (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)), 
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,  
    maxEpochsVal::Int=20, showText::Bool=false)

    inputs, targets = trainingDataset
    numInputs = size(inputs, 2)
    numOutputs = 1  # For two-class case

    ann = buildClassANN(numInputs, topology, numOutputs; transferFunctions=transferFunctions)
    loss(m, x, y) = Flux.Losses.binarycrossentropy(m(x), y)

    opt_state = Flux.setup(Flux.Adam(learningRate), ann)

    train_loss_history = Float64[]
    val_loss_history = Float64[]
    test_loss_history = Float64[]

    x_train = inputs'
    y_train = reshape(targets, 1, :)

    x_val, y_val = validationDataset
    x_val = x_val'
    y_val = reshape(y_val, 1, :)

    x_test, y_test = testDataset
    x_test = x_test'
    y_test = reshape(y_test, 1, :)

    # Initial loss (cycle 0)
    push!(train_loss_history, loss(ann, x_train, y_train))
    if size(x_val,2) > 0
        push!(val_loss_history, loss(ann, x_val, y_val))
    end
    if size(x_test,2) > 0
        push!(test_loss_history, loss(ann, x_test, y_test))
    end

    best_val_loss = Inf
    best_ann = deepcopy(ann)
    epochs_since_best = 0

    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(x_train, y_train)], opt_state)
        train_loss = loss(ann, x_train, y_train)
        push!(train_loss_history, train_loss)

        # Validation loss
        val_loss = nothing
        if size(x_val,2) > 0
            val_loss = loss(ann, x_val, y_val)
            push!(val_loss_history, val_loss)
            if val_loss < best_val_loss
                best_val_loss = val_loss
                best_ann = deepcopy(ann)
                epochs_since_best = 0
            else
                epochs_since_best += 1
            end
        end

        # Test loss
        if size(x_test,2) > 0
            test_loss = loss(ann, x_test, y_test)
            push!(test_loss_history, test_loss)
        end

        # Early stopping
        if train_loss <= minLoss
            break
        end
        if size(x_val,2) > 0 && epochs_since_best >= maxEpochsVal
            break
        end
    end

    # Return best ANN if validation set was used, else last ANN
    if size(x_val,2) > 0
        return best_ann, train_loss_history, val_loss_history, test_loss_history
    else
        return ann, train_loss_history, val_loss_history, test_loss_history
    end
end

# holdOut from pract 3

using Random

function holdOut(N::Int, P::Real)
    @assert 0.0 <= P <= 1.0
    n_test = round(Int, N * P)
    indices = randperm(N)
    test_idx = indices[1:n_test]
    train_idx = indices[(n_test+1):end]
    return (train_idx, test_idx)
end

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    @assert 0.0 <= Pval <= 1.0 && 0.0 <= Ptest <= 1.0 && Pval + Ptest <= 1.0 "Invalid proportions"
    trainAux, test = holdOut(N, Ptest)
    PvalAdjusted = (Pval * N) / length(trainAux)
    valIdx, trainIdx = holdOut(length(trainAux), PvalAdjusted)
    validate = trainAux[valIdx]
    train = trainAux[trainIdx]

    return (train, validate, test)
end


# confusionMatrix from pract 4.2

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)

    # first we get the number of columns of the matrix, to know with how many classes we are working
    numClasses = size(outputs, 2)

    if numClasses == 2
        error("Invalid number of classes since it is equivalent to do a binary classification.")
    end

    # first, we get the accuracy and the error rate
    correct = sum(outputs .& targets) + sum((.!outputs) .& (.!targets))
    total = length(outputs)
    accuracy = correct / total
    errorRate = 1 - accuracy

    # then, we save memory for the different metrics of each class
    sensitivityList = zeros(numClasses)
    specificityList = zeros(numClasses)
    posPredValList = zeros(numClasses)
    negPredValList = zeros(numClasses)
    fScoreList = zeros(numClasses)

    for c in 1:numClasses
        predicted = outputs[:, c]
        actual = targets[:, c]

        truePos = sum(predicted .& actual)
        falsePos = sum(predicted .& .!actual)
        falseNeg = sum((.!predicted) .& actual)
        trueNeg = sum((.!predicted) .& .!actual)

        # Check for special cases
        allTrueNeg = (truePos == 0 && falsePos == 0 && falseNeg == 0 && trueNeg > 0)
        allTruePos = (truePos > 0 && falsePos == 0 && falseNeg == 0 && trueNeg == 0)

        # sensitivity or recall
        if allTrueNeg
            sensitivityList[c] = 1.0
        elseif (truePos + falseNeg) == 0
            sensitivityList[c] = 0.0
        else
            sensitivityList[c] = truePos / (truePos + falseNeg)
        end

        # specificity
        if allTruePos
            specificityList[c] = 1.0
        elseif (trueNeg + falsePos) == 0
            specificityList[c] = 0.0
        else
            specificityList[c] = trueNeg / (trueNeg + falsePos)
        end


        # positive predictive value or precision
        if allTrueNeg
            posPredValList[c] = 1.0
        elseif (truePos + falsePos) == 0
            posPredValList[c] = 0.0
        else
            posPredValList[c] = truePos / (truePos + falsePos)
        end

        # negative predictive value
        if allTruePos
            negPredValList[c] = 1.0
        elseif (trueNeg + falseNeg) == 0
            negPredValList[c] = 0.0
        else
            negPredValList[c] = trueNeg / (trueNeg + falseNeg)
        end

        # F1-Score
        if sensitivityList[c] == 0.0 && posPredValList[c] == 0.0
            fScoreList[c] = 0.0
        else
            fScoreList[c] = 2 * (posPredValList[c] * sensitivityList[c]) / (posPredValList[c] + sensitivityList[c])
        end
    end

    confusionMatrix = [sum((targets[:, i] .== 1) .& (outputs[:, j] .== 1)) for i in 1:numClasses, j in 1:numClasses]

    if weighted
        samplesPerClass = vec(sum(targets, dims=1))
        weights = samplesPerClass ./ sum(samplesPerClass)
        sensitivity = sum(weights .* sensitivityList)
        specificity = sum(weights .* specificityList)
        posPredVal = sum(weights .* posPredValList)
        negPredVal = sum(weights .* negPredValList)
        fScore = sum(weights .* fScoreList)
    else
        sensitivity = mean(sensitivityList)
        specificity = mean(specificityList)
        posPredVal = mean(posPredValList)
        negPredVal = mean(negPredValList)
        fScore = mean(fScoreList)
    end


    return accuracy, errorRate, sensitivity, specificity, posPredVal, negPredVal, fScore, confusionMatrix
end

function confusionMatrix(outputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    outputsBool = classifyOutputs(outputs, threshold)
    return confusionMatrix(outputsBool, targets; weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    # first, we do the assertion to check that all values in outputs and targets are in classes
    @assert all(x -> x in classes, unique(outputs)) && all(x -> x in classes, unique(targets))

    # then we do the oneHotEncoding and call the confusionMatrix function for boolean matrices
    oneHotOutputs = oneHotEncoding(outputs, classes)
    oneHotTargets = oneHotEncoding(targets, classes)

    return confusionMatrix(oneHotOutputs, oneHotTargets; weighted=weighted)
end


function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # we generate the classes vector from the unique values in outputs and targets
    classes = unique(vcat(targets, outputs))

    # and then we call the previous function
    return confusionMatrix(outputs, targets, classes; weighted=weighted)
end


function ANNCrossValidation(topology::AbstractArray{<:Int,1},
        dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
        crossValidationIndices::Array{Int64,1};
        numExecutions::Int=50,
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        validationRatio::Real=0, maxEpochsVal::Int=20)
    #TODO
    # 1- we get the classes via the targets
    classes = sort(unique(dataset[2]))

    # 2- then we do a one-hot encoding
    classesEncoded= oneHotEncoding(classes)

    # 3- we get the number of numFolds
    numFolds = maximum(crossValidationIndices)

    # 4- we create the vectors for the metrics to store the information of each fold
    # to do so properly, first we must get the number of classes
    numClasses = length(classes)

    accuracyList = zeros(numFolds)
    errorRate = zeros(numFolds)

    sensitivityList = zeros(numClasses, numFolds)
    specificityList = zeros(numClasses, numFolds)
    posPredValList = zeros(numClasses, numFolds)
    negPredValList = zeros(numClasses, numFolds)
    fScoreList = zeros(numClasses, numFolds)

    # 5- we prepare the variable to store the accumulation of the confussion matrix
    globalConfusionMatrix = zeros(numClasses, numClasses, numFolds)

    # we get the partitions
    listFoldData = []
    # findall(x .== fold)
    for fold in 1:numFolds
        listCurrentFold = findall(x -> x == fold, crossValidationIndices)
        dataCurrentFold = (dataset[1][listCurrentFold, :], dataset[2][listCurrentFold])
        push!(listFoldData, dataCurrentFold)
    end
    
    # and then, we start the execution of the folds
    for fold in 1:numFolds
        # first, we prepare the space for the metrics (per execution)
        perExecutionAccuracy = zeros(numExecutions)
        perExecutionErrorRate = zeros(numExecutions)

        perExecutionSensitivityList = zeros(numClasses, numExecutions)
        perExecutionSpecificityList = zeros(numClasses, numExecutions)
        perExecutionPosPredValList = zeros(numClasses, numExecutions)
        perExecutionNegPredValList = zeros(numClasses, numExecutions)
        perExecutionFScoreList = zeros(numClasses, numExecutions)

        perExecutionConfMatrix = zeros(numClasses, numClasses, numExecutions)
        
        # then we get the test dataset, as it is just one fold, which can be the current one
        testDataset = listFoldData[fold]

        # in case there's a validation ratio, we must obtain the division between training and validation using holdOut
        if validationRatio > 0
            train, validate = holdOut(numFolds-1, validationRatio)
            for i in eachindex(train)
                if train[i] >= fold
                    train[i] += 1
                end
            end
            
            for i in eachindex(validate)
                if validate[i] >= fold
                    validate[i] += 1
                end
            end
            
            trainingX = vcat([listFoldData[i][1] for i in train]...)
            trainingY = vcat([listFoldData[i][2] for i in train]...)
            trainingDataset = (trainingX, trainingY)

            validationX = vcat([listFoldData[i][1] for i in validate]...)
            validationY = vcat([listFoldData[i][2] for i in validate]...)
            validationDataset = (validationX, validationY)
        
        # otherwise, everything else other than the current test fold goes to the training set
        else
            # we get the index that are not test
            train = filter(i -> i != fold, 1:numFolds)

            trainingX = vcat([listFoldData[i][1] for i in train]...)
            trainingY = vcat([listFoldData[i][2] for i in train]...)
            trainingDataset = (trainingX, trainingY)
            validationDataset = (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0))

        end
        
        # and now that we have the sets, we start doing the amount of expected executions (because ann are indeterministic)
        # and then we obtain the metrics per execution
        for j in 1:numExecutions
            ann, _, _, _ = trainClassANN(topology, trainingDataset, validationDataset, testDataset, transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal, false)
            x_test, y_test = testDataset
            y_pred = ann(x_test')
            perExecutionAccuracy[j], perExecutionErrorRate[j], perExecutionSensitivityList[:, j], perExecutionSpecificityList[:, j], perExecutionPosPredValList[:, j], perExecutionNegPredValList[:, j], perExecutionFScoreList[:, j], perExecutionConfMatrix[:, :, j] = confusionMatrix(y_pred', y_test)
        end
        # and, after all the execution, we introduce the mean into the external lists
        accuracyList[fold] = mean(perExecutionAccuracy)
        errorRate[fold] = mean(perExecutionErrorRate)

        sensitivityList[:, fold] = dropdims(mean(perExecutionSensitivityList, dims=2); dims=2)
        specificityList[:, fold] = dropdims(mean(perExecutionSpecificityList, dims=2); dims=2)
        posPredValList[:, fold] = dropdims(mean(perExecutionPosPredValList, dims=2); dims=2)
        negPredValList[:, fold] = dropdims(mean(perExecutionNegPredValList, dims=2); dims=2)
        fScoreList[:, fold] = dropdims(mean(perExecutionFScoreList, dims=2); dims=2)
        
        globalConfusionMatrix[:,:,fold] = dropdims(mean(perExecutionConfMatrix, dims=3); dims=3)
    end
    #now, after the loop, this are the following returned data:
    """ 1. Accuracy (mean, std)
        2. Error rate (mean, std)
        3. Sensitivity (mean, std)
        4. Specificity (mean, std)
        5. PPV (mean, std)
        6. NPV (mean, std)
        7. F1-score (mean, std)
        8. Global test confusion matrix (which is basically the mean?)"""
    return ((mean(accuracyList), std(accuracyList)), (mean(errorRate), std(errorRate)),
    (dropdims(mean(sensitivityList, dims=2); dims=2), dropdims(std(sensitivityList, dims=2); dims=2)),
    (dropdims(mean(specificityList, dims=2); dims=2), dropdims(std(specificityList, dims=2); dims=2)),
    (dropdims(mean(posPredValList, dims=2); dims=2), dropdims(std(posPredValList, dims=2); dims=2)),
    (dropdims(mean(negPredValList, dims=2); dims=2), dropdims(std(negPredValList, dims=2); dims=2)),
    (dropdims(mean(fScoreList, dims=2); dims=2), dropdims(std(fScoreList, dims=2); dims=2)),   
    dropdims(sum(globalConfusionMatrix, dims=3); dims=3))

end