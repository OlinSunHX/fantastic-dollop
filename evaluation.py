import numpy as np
import torch

def evaluate_metrics(ground_truth, predictions, metrics, verbose=True, print_tex=True):
    results = {}
    for name, metric in metrics.items():
        # print("ground_truth:",ground_truth)
        # print("predictions:", predictions)
        results[name] = metric(ground_truth, predictions)
        # print("results:", results)
    if verbose:
        pass
        # print(', '.join(f'{name}={results[name]:.2f}' for name in metrics))
    elif print_tex:
        tex = ' & '.join(f'{results[name]:.2f}' for name in metrics)
        print(tex)

    return results

# Test loop
def evaluate_single(net, dataloader, device, metrics_valence_arousal=None, metrics_expression=None, metrics_au=None,   verbose=True, print_tex=False):
    net.eval()

    for index, data in enumerate(dataloader):
        # print("index:",index)
        images = data['image'].to(device)
        valence = data.get('valence', None)
        arousal = data.get('arousal', None)
        expression = data.get('expression', None)

        print("images:", np.shape(images))
        # print("valence:", np.shape(valence),valence)
        # print("arousal:", np.shape(arousal),arousal)
        # print("expression:", np.shape(expression), expression)

        with torch.no_grad():
            out = net(images)

        # shape_pred = out['heatmap']
        if expression is not None:
            expr = out['expression']
            # expr=torch.nn.softmax()
            print(expr) #tensor([[-2.7500, -3.1039, -1.9960, -3.2175, -2.1978]])
            expr = np.argmax(np.squeeze(expr.cpu().numpy()))
            print(expr) #2
            # expr = np.argmax(np.squeeze(expr.cpu().numpy()), axis=1)

        if index:
            if expression is not None:
                expression_pred = np.concatenate([expr, expression_pred])
                expression_gts = np.concatenate([expression, expression_gts])
        else:
            if expression is not None:
                expression_pred = expr
                expression_gts = expression
    # expression_gts=expression_gts.numpy()
    # print("ground_truth:", expression_gts)
    # print("predictions:", expression_pred)
    ##0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt'
    pred_label = []
    if expression_pred == 0:
        pred_label.append("neutral")
    elif expression_pred == 1:
        pred_label.append("happy")
    elif expression_pred == 2:
        pred_label.append("sad")
    elif expression_pred == 3:
        pred_label.append("surprise")
    elif expression_pred == 4:
        pred_label.append("fear")
    elif expression_pred == 5:
        pred_label.append("disgust")
    elif expression_pred == 6:
        pred_label.append("anger")
    elif expression_pred == 7:
        pred_label.append("contempt")
    print("预测的人脸表情结果为：", expression_pred)
    print("预测的人脸表情标签为：", pred_label)

def evaluate_multiple(net, dataloader, device, metrics_valence_arousal=None, metrics_expression=None, metrics_au=None,  verbose=True, print_tex=False):
    net.eval()

    for index, data in enumerate(dataloader):
        # print("index:",index)
        images = data['image'].to(device)
        valence = data.get('valence', None)
        arousal = data.get('arousal', None)
        expression = data.get('expression', None)

        print("images:", np.shape(images))
        # print("valence:", np.shape(valence),valence)
        # print("arousal:", np.shape(arousal),arousal)
        # print("expression:", np.shape(expression), expression)

        with torch.no_grad():
            out = net(images)

        # shape_pred = out['heatmap']
        if expression is not None:
            expr = out['expression']
            # print(expr)
            expr = np.argmax(np.squeeze(expr.cpu().numpy()), axis=1)
            # print(expr)

        if index:
            print("index:", index)
            if expression is not None:
                expression_pred = np.concatenate([expr, expression_pred])
                expression_gts = np.concatenate([expression, expression_gts])
        else:
            if expression is not None:
                expression_pred = expr
                expression_gts = expression
    pred_label = []
    # expression_gts=expression_gts.numpy()
    # print("ground_truth:", expression_gts)
    # print("predictions:", expression_pred)
    ##0: 'neutral', 1:'happy', 2:'sad', 3:'surprise', 4:'fear', 5:'disgust', 6:'anger', 7:'contempt'
    for i  in range(len(expression_pred)):
        if expression_pred[i]==0:
            pred_label.append("neutral")
        elif expression_pred[i]==1:
            pred_label.append("happy")
        elif expression_pred[i]==2:
            pred_label.append("sad")
        elif expression_pred[i]==3:
            pred_label.append("surprise")
        elif expression_pred[i]==4:
            pred_label.append("fear")
        elif expression_pred[i]==5:
            pred_label.append("disgust")
        elif expression_pred[i]==6:
            pred_label.append("anger")
        elif expression_pred[i] == 7:
            pred_label.append("contempt")
    print("预测的人脸表情结果为：", expression_pred)
    print("预测的人脸表情标签为：", pred_label)

def evaluate(net, dataloader, device, metrics_valence_arousal=None, metrics_expression=None, metrics_au=None, verbose=True, print_tex=False):
    
    net.eval()

    for index, data in enumerate(dataloader):
        # print("index:",index)
        images = data['image'].to(device)
        valence = data.get('valence', None)
        arousal = data.get('arousal', None)
        expression = data.get('expression', None)

        print("images:", np.shape(images))
        # print("valence:", np.shape(valence),valence)
        # print("arousal:", np.shape(arousal),arousal)
        # print("expression:", np.shape(expression), expression)

        with torch.no_grad():
            out = net(images)
      
        #shape_pred = out['heatmap']
        if expression is not None:
            expr = out['expression']
            # print(expr)
            expr = np.argmax(np.squeeze(expr.cpu().numpy()), axis=1)
            # print(expr)
            #

        if metrics_valence_arousal is not None:
            val = out['valence']
            ar = out['arousal']

            val = np.squeeze(val.cpu().numpy())
            ar = np.squeeze(ar.cpu().numpy())

        if index:
            # print("index:",index)
            if metrics_valence_arousal is not None:
                valence_pred = np.concatenate([val, valence_pred])
                arousal_pred = np.concatenate([ar,  arousal_pred])
                valence_gts = np.concatenate([valence, valence_gts])
                arousal_gts = np.concatenate([arousal,  arousal_gts])
        
            if expression is not None:
                # print("expr0:",expr)
                # print("expression_pred:",expression_pred)
                # print("expression:", expression)
                # print("expression_gts:", expression_gts)
                expression_pred = np.concatenate([expr, expression_pred])
                expression_gts = np.concatenate([expression, expression_gts])
                # print("expr1:", expr)
                # print("expression_pred:", expression_pred)
                # print("expression:", expression)
                # print("expression_gts:", expression_gts)
        else:
            if metrics_valence_arousal is not None:
                valence_pred = val
                arousal_pred = ar
                valence_gts = valence
                arousal_gts = arousal

            if expression is not None:    
                expression_pred = expr
                expression_gts = expression
            
    if metrics_valence_arousal is not None:
        #Clip the predictions
        valence_pred = np.clip(valence_pred, -1.0,1.0)
        arousal_pred = np.clip(arousal_pred, -1.0,1.0)

        #Squeeze if valence_gts is shape (N,1)
        valence_gts = np.squeeze(valence_gts)
        arousal_gts = np.squeeze(arousal_gts)

    if metrics_expression is not None:
        if verbose:
            print('\nExpression')
        acc_expressions = evaluate_metrics(expression_gts, expression_pred, metrics=metrics_expression, verbose=verbose, print_tex=print_tex)

    if metrics_valence_arousal is not None:
        if verbose:
            print('\nValence')
        valence_results = evaluate_metrics(valence_gts, valence_pred, metrics=metrics_valence_arousal, verbose=verbose, print_tex=print_tex)
        if verbose:
            print('Arousal')
        arousal_results = evaluate_metrics(arousal_gts, arousal_pred, metrics=metrics_valence_arousal, verbose=verbose, print_tex=print_tex)

    net.train()    
    
    #Return the correct amount of parameters depending on the type of evaluation
    if metrics_expression is not None:
        if metrics_valence_arousal is not None:
            return valence_results, arousal_results, acc_expressions
        else:
            return acc_expressions
    else:
            return valence_results, arousal_results

def evaluate_flip(net, dataloader_no_flip, dataloader_flip, device, metrics_valence_arousal=None, metrics_expression=None, metrics_au=None, verbose=True, print_tex=False):
    
    net.eval()

    #Loop without flip
    for index, data in enumerate(dataloader_no_flip):
        images = data['image'].to(device)
        valence = data.get('valence', None)
        arousal = data.get('arousal', None)
        expression = data.get('expression', None)

        print("images:",np.shape(images))
        # print("valence:", np.shape(valence),valence)
        # print("arousal:", np.shape(arousal),arousal)
        # print("expression:", np.shape(expression),expression)

        with torch.no_grad():
            out = net(images)
      
        #shape_pred = out['heatmap']
        if expression is not None:
            expr = out['expression']
            expr = np.argmax(np.squeeze(expr.cpu().numpy()), axis=1)

        if metrics_valence_arousal is not None:
            val = out['valence']
            ar = out['arousal']

            val = np.squeeze(val.cpu().numpy())
            ar = np.squeeze(ar.cpu().numpy())

        if index:
            if metrics_valence_arousal is not None:
                valence_pred = np.concatenate([valence_pred, val])
                arousal_pred = np.concatenate([arousal_pred, ar])
                valence_gts = np.concatenate([valence_gts, valence])
                arousal_gts = np.concatenate([arousal_gts, arousal])
        
            if expression is not None:        
                expression_pred = np.concatenate([expression_pred, expr])
                expression_gts = np.concatenate([expression_gts, expression])
        else:
            if metrics_valence_arousal is not None:
                valence_pred = val
                arousal_pred = ar
                valence_gts = valence
                arousal_gts = arousal

            if expression is not None:    
                expression_pred = expr
                expression_gts = expression
    
    valence_pred = valence_pred.astype(np.float64)
    arousal_pred = arousal_pred.astype(np.float64)
    valence_gts = valence_gts.astype(np.float64)
    arousal_gts = arousal_gts.astype(np.float64)

    #Loop with flip
    n_images = 0
    for index, data in enumerate(dataloader_flip):
        images = data['image'].to(device)
        valence = data.get('valence', None)
        arousal = data.get('arousal', None)
        expression = data.get('expression', None)
        
        with torch.no_grad():
            out = net(images)
      
        #shape_pred = out['heatmap']
        #if expression is not None:
        #    expr = out['expression']
        #    expr = np.argmax(np.squeeze(expr.cpu().numpy()), axis=1)

        if metrics_valence_arousal is not None:
            val = out['valence']
            ar = out['arousal']

            val = np.squeeze(val.cpu().numpy()).astype(np.float64)
            ar = np.squeeze(ar.cpu().numpy()).astype(np.float64)

        for k in range(0, images.size(0)):
            valence_pred[n_images] = (val[k]+valence_pred[n_images])/2.0
            arousal_pred[n_images] = (ar[k]+arousal_pred[n_images])/2.0
            n_images += 1
    
    if metrics_valence_arousal is not None:
        #Clip the predictions
        valence_pred = np.clip(valence_pred, -1.0,1.0)
        arousal_pred = np.clip(arousal_pred, -1.0,1.0)

        #Squeeze if valence_gts is shape (N,1)
        valence_gts = np.squeeze(valence_gts)
        arousal_gts = np.squeeze(arousal_gts)

    if metrics_expression is not None:
        if verbose:
            print('\nExpression')
        acc_expressions = evaluate_metrics(expression_gts, expression_pred, metrics=metrics_expression, verbose=verbose, print_tex=print_tex)

    if metrics_valence_arousal is not None:
        if verbose:
            print('\nValence')
        valence_results = evaluate_metrics(valence_gts, valence_pred, metrics=metrics_valence_arousal, verbose=verbose, print_tex=print_tex)
        if verbose:
            print('Arousal')
        arousal_results = evaluate_metrics(arousal_gts, arousal_pred, metrics=metrics_valence_arousal, verbose=verbose, print_tex=print_tex)

    net.train()    
    
    #Return the correct amount of parameters depending on the type of evaluation
    if metrics_expression is not None:
        if metrics_valence_arousal is not None:
            return valence_results, arousal_results, acc_expressions
        else:
            return acc_expressions
    else:
            return valence_results, arousal_results
