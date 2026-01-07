import numpy
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
def validate(n_classes, model, loader):
    confusionmatrix = numpy.zeros((n_classes, n_classes))
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(loader):
            images_batch, target_labels = \
                sample_batched[0], sample_batched[1]
            images_batch, target_labels = images_batch.cuda(), target_labels.cuda()
            #print(images_batch.shape,target_labels.shape)
            x = model(images_batch)
            target_labels = target_labels.cpu().numpy().astype(numpy.uint8)
            probs = F.softmax(x, dim=1)
            m_label_out = probs.max(1)[1].cpu().numpy().astype(numpy.uint8)
            confusionmatrix += confusion_matrix(target_labels.flatten(), m_label_out.flatten(),
                                                    labels=range(n_classes))

    acc = numpy.diag(confusionmatrix).sum() / confusionmatrix.sum()
    mean_acc = numpy.nanmean(acc)
    iu = numpy.diag(confusionmatrix) / (confusionmatrix.sum(axis=1) + confusionmatrix.sum(axis=0) - numpy.diag(confusionmatrix))
    mean_iu = numpy.nanmean(iu)
    recall = numpy.diag(confusionmatrix) / confusionmatrix.sum(axis=0)
    precision = numpy.diag(confusionmatrix) / confusionmatrix.sum(axis=1)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # print("iou:{:.6f}".format(mean_iu))
    return mean_acc, iu[1],recall[1],precision[1],f1_score[1]
