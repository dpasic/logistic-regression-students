function g = sigmoid(z)
    % compute the sigmoid of each value of z
    g = 1 ./ (1 + e.^-z);
end