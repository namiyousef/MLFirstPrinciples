def generalisation_error(
        model,
        data_generator,
        k_range=[1, 50],
        runs=100,
        test_size=0.2,
        seed=None
):
    """
    Determines the generalisation error as a function of k

    model : class
        Class of the model to use. Must be built using sklearn API

    data_generator : tuple
        tuple in the form (generator_function, parameters)

    k_range : iterable
        defines the range of k's to use for the problem. Defaults to [1,50)

    runs : int
        indicates how many times to repeat the experiment

    test_size : float
        indicates percentage of train set to use for testing

    """
    data_generator, parameters = data_generator

    global_errors = []
    tot_time = 0
    for i, k_ in enumerate(k_range):
        start = time()
        print(f'We are at k={k_}')
        tot_err = 0
        for run in range(runs):
            X, y = data_generator(**parameters)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            kNN_model = model(k_)
            kNN_model.fit(X_train, y_train)

            y_hat = kNN_model.predict(
                X_test.astype(np.float32)
                # np.ascontiguousarray(X_test),
            )

            correctness_index = [y_hat_ == y_test_ for y_hat_, y_test_ in zip(y_hat, y_test)]

            tot_err += (len(correctness_index) - sum(correctness_index)) / len(correctness_index)

        global_errors.append(tot_err / runs)
        time_taken = time() - start
        tot_time += time_taken
        print(
            f'Time taken for k={k_} is {time_taken}. Estimated time remaining is {(tot_time / (i + 1)) * (len(k_range) - i - 1)}')

        # average the generalisation error

    return global_errors