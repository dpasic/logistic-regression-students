function x_hypothesis = add_hypothesis_column(x)
    % number of examples
    m = size(x, 1);

    % add a column (containing ones) at the beginning of the matrice x
    % will be used for computing hypothesis
    x_hypothesis = [ones(m, 1), x];
end