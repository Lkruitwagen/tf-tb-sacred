


@ex.capture
def train(logger, pbar_len, n_epochs, model, optimizer, trn_generator, val_generator, writer):

    for epoch in range(n_epochs):
        print("\nStart of epoch %d" % (epoch,))
        
        ### create our metrics to track:
        train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        val_acc_metric = tf.keras.metrics.CategoricalAccuracy()
        
        ### create a progress to track training progess
        pbar = tqdm(total=pbar_len)

        # Iterate over the batches of the dataset.
        for step, (X_batch_trn, Y_batch_trn) in enumerate(tqdm(trn_generator)):

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                Y_hat_trn = model(X_batch_trn, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                loss_value = loss_fn(Y_batch_trn, Y_hat_trn)

            # Use the gradient tape to automatically retrieve the gradients
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            # update the training metrics
            train_acc_metric.update_state(Y_batch_trn, Y_hat_trn)
            
            pbar.update(1)
            
        ### do eval loop
        for X_batch_val, Y_batch_val in val_dataset:
            Y_hat_val = model(X_batch_val, training=False)
            # Update val metrics
            val_acc_metric.update_state(Y_batch_val, Y_hat_val)
            
            pbar.update(1)

            
        pbar.close()
        
        ### write our tensorboard variables
        trn_acc = trn_acc_metric.result()
        val_acc = val_acc_metric.result()
        
        logger.info(f'training accuray: {trn_acc}; validation accuracy: {val_acc}')
        
        ### on epoch end
        trn_acc_metric.reset_states()
        val_acc_metric.reset_states()
        