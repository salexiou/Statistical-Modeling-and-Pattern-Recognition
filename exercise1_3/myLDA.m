function A = myLDA(Samples, Labels, NewDim)
% Input:
%   Samples: The Data Samples
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA

	  [NumSamples ,NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if(NumSamples ~= NumLabels) then
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes)  %The number of classes

    % Initialize Sw and Sb matrices
    Sw = zeros(NumFeatures, NumFeatures);
    Sb = zeros(NumFeatures, NumFeatures);
    %For each class i
    % Initialize variables
    Sw = zeros(NumFeatures, NumFeatures);
    Sb = zeros(NumFeatures, NumFeatures);
    P = zeros(1, NumClasses);
    mu = zeros(NumClasses, NumFeatures);

	%Find the necessary statistics
    for i = 1:NumClasses
        %Calculate the Class Prior Probability
      P(i)= sum(Labels == Classes(i))/NumClasses ;
        %Calculate the Class Mean
      mu(i,:)= mean(Samples(Labels == Classes(i),:));
        %Calculate the Within Class Scatter Matrix
      Sw = Sw + P(i)*cov(Samples(Labels == Classes(i),:));
        %Calculate the Global Mean
      m0= mean(Samples) ;


        %Calculate the Between Class Scatter Matrix
      Sb = Sb + P(i) * (mu(i,:) - m0)*transpose(mu(i,:) - m0);
    end
    %Calculate the Class Prior Probability


    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;

    %Perform Eigendecomposition
    [U,S] = svd(EigMat);


    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
	%% You need to return the following variable correctly.
	A=U(:,1:NewDim);  % Return the LDA projection vectors
end
