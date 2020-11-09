function [error_train, error_cross] = learning_curve(x_train, y_train, x_cross, y_cross, lambda)
    % number of training examples
    m = size(x_train, 1);
    % number of features
    n = size(x_train, 2);

    error_train = zeros(m, 1);
    error_cross = zeros(m, 1);

    for i = 1:m,
        x_train_curve = x_train(1:i, :);
        y_train_curve = y_train(1:i);
        theta_curve = zeros(n, 1);
        theta_curve = gradient_descent(x_train_curve, y_train_curve, theta_curve, lambda);
        
        error_train(i) = cost_function(x_train_curve, y_train_curve, theta_curve, lambda);
        % set the previous value if the current one is NaN
        if isnan(error_train(i)),
            error_train(i) = error_train(i - 1);
        end;
        
        error_cross(i) = cost_function(x_cross, y_cross, theta_curve, lambda);
        % set the previous value if the current one is NaN
        if isnan(error_cross(i)),
            error_cross(i) = error_cross(i - 1);
        end;
    end;
end