def plot_cm(y_test, y_pred, title):
    
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Purples)
    ax.figure.colorbar(im, ax=ax)

    classes = ['dMMR', 'pMMR']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f'
    fmt2 = 'd'
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt2)+"\n"+format(cm_normalized[i, j], fmt),
                    ha="center", va="center", size="large",
                    color="white" if cm_normalized[i, j] > thresh else "black")

    plt.title(title)
    plt.show()
    
    
def data(msis, msss, root, froot):
    
    msisdone, msssdone = [], []

    for i in range(len(msis)):
        p5x = np.load(root+"probs5x/"+msis[i]+".npy")
        p5x = p5x[:,0]
        p5x = list(p5x)
        msisdone.append(msis[i])

    for i in range(len(msss)):
        p5x = np.load(root+"probs5x/"+msss[i]+".npy")
        p5x = p5x[:,0]
        p5x = list(p5x)
        msssdone.append(msss[i])

    imgs = msisdone + msssdone

    weight = len(msisdone)/len(msssdone)
    weights0 = np.full(len(msisdone),1)
    weights1 = np.full(len(msssdone),weight)
    sample_weights = np.hstack((weights0, weights1))
    
    print(len(msisdone))
    print(len(msssdone))
    labels = len(msisdone)*[0]+len(msssdone)*[1]

    data = np.load(froot+imgs[0]+"FEATURES.npy")

    for i in range(1, len(imgs)):
        row = np.load(froot+imgs[i]+"FEATURES.npy")
        data = np.vstack((data,row))
    
    data5x = data[:,:13]
    data20x = data[:,13:]

    return data5x, data20x, labels, sample_weights

def predict(data5x, data20x, labels, weights, root, froot, idx, name, bst5x, bst20x):

    y_pred = bst5x.predict(data5x)
    probas = bst5x.predict_proba(data5x)
    probas5x = list(probas[:,0])

    y_pred = bst20x.predict(data20x)
    probas = bst20x.predict_proba(data20x)
    probas20x = list(probas[:,0])

    predsens = []
    
    for i in range(len(probas5x)):
        val1 = probas5x[i]
        val2 = probas20x[i]
        pred = np.mean([val1, val2])
        predsens.append(pred)
    
    y_pred = []

    for i in range(len(predsens)):
        if (predsens[i]>th):
            y_pred.append(0)
        if (predsens[i]<=th):
            y_pred.append(1)

    y_pred = np.array(y_pred)
    labels = np.array(labels)

    tp = np.sum((y_pred == 0) & (labels == 0))
    tn = np.sum((y_pred == 1) & (labels == 1))
    fp = np.sum((y_pred == 0) & (labels == 1))
    fn = np.sum((y_pred == 1) & (labels == 0))

    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    
    NPV = tn / (tn+fn)
    print("NPV: " +str(NPV))

    fpr, tpr, _ = metrics.roc_curve(labels, y_pred, pos_label =0)
    auc = metrics.roc_auc_score(labels, y_pred, sample_weight = weights)
    prauc = metrics.average_precision_score(labels, predsens, pos_label=0, sample_weight = weights)
    praucs.append(round(prauc,3))
    f1ens = metrics.f1_score(labels, y_pred, pos_label=0, sample_weight=weights, average='binary')
    
    print(prauc)
    print()
    print("F-score: " + str(round(f1ens, 4)))
    print("Sensitivity: " + str(round(sensitivity, 4)))
    print("Specificity: " + str(round(specificity, 4)))
    print("AUC: " + str(round(auc, 4)))
    print("PR AUC: " + str(round(prauc, 4)))

    cl=Counter(labels.ravel())[0] / labels.size
    cl = round(cl, 3)

    metrics.PrecisionRecallDisplay.from_predictions(labels, predsens, sample_weight=weights, pos_label=0, ax=ax, name="multi-scale", label="multi-scale: "+" (AUCPR="+str(prauc+")")
    plt.title(name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.show()
    
    plot_cm(labels, y_pred, "multi-scale")
