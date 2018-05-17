# Feed-Forward-Neural-Network-for-modularity-classification-of-integers

Classyfies array of 3 integers, output is 0 if all are odd, 1 if even and range in between can be seen as a confidence score of
which group it most likely belongs to. Uses feature scaling to bound possible input/Output values to the closed interval [0,1]. Due to correlation being reduced to the possible combinations of [m,kn,n],  where m < kn < n, && n >= 1. Relative positioning of integers in array is eliminated as possible correlation point through randomization of the size of numbers positioned in them. Increasing score should therefore be a matter of providing more datapoints. Testing case written at the bottom of the code is [100,120,140], which can seen as a special case of [k,6/7n,n], where n = 140. Will most likely be generalized to allow arbitrary devisors and number of groups to classify.  
