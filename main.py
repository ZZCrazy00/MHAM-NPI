from model import *
from utile import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

batch_size = 1024
embedding_size = 2
data_name = 'NPInter2'

sparse_feature = ['R' + str(i) for i in range(1, 257)]
dense_feature = ['P' + str(i) for i in range(1, 401)]
col_names = ['label'] + dense_feature + sparse_feature
data = pd.read_csv('data/{}/sample.txt'.format(data_name), names=col_names, sep='\t')

data[sparse_feature] = data[sparse_feature].fillna('-1', )
data[dense_feature] = data[dense_feature].fillna('0', )
target = ['label']

feat_sizes = {}
feat_sizes_dense = {feat: len(data[feat].unique()) for feat in dense_feature}
feat_sizes_sparse = {feat: len(data[feat].unique()) for feat in sparse_feature}
feat_sizes.update(feat_sizes_dense)
feat_sizes.update(feat_sizes_sparse)

for feat in sparse_feature:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
nms = MinMaxScaler(feature_range=(0, 1))
data[dense_feature] = nms.fit_transform(data[dense_feature])

fixlen_feature_columns = [(feat, 'sparse') for feat in sparse_feature] + [(feat, 'dense') for feat in dense_feature]
dnn_feature_columns = fixlen_feature_columns

train, test = train_test_split(data, test_size=0.2, random_state=2022)

train_label = pd.DataFrame(train['label'])
train = train.drop(columns=['label'])
train_tensor_data = TensorDataset(torch.from_numpy(np.array(train)), torch.from_numpy(np.array(train_label)))
train_loader = DataLoader(train_tensor_data, shuffle=False, batch_size=batch_size)

test_label = pd.DataFrame(test['label'])
test = test.drop(columns=['label'])
test_tensor_data = TensorDataset(torch.from_numpy(np.array(test)), torch.from_numpy(np.array(test_label)))
test_loader = DataLoader(test_tensor_data, batch_size=batch_size)

model = MHANN(feat_sizes, embedding_size, dnn_feature_columns).cuda()
loss_func = nn.BCELoss(reduction='mean').cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)

epoches = 200
for epoch in range(epoches):
    total_loss_epoch = 0.0
    total_tmp = 0
    model.train()
    for index, (x, y) in enumerate(train_loader):
        x, y = x.cuda().float(), y.cuda().float()
        y_hat = model(x)

        optimizer.zero_grad()
        loss = loss_func(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
        total_tmp += 1
    auc, sen, pre, F1 = get_result(test_loader, model)
    print('epoch/epoches: {}/{}, train loss: {:.5f}, '
          'test auc: {:.3f}, sen: {:.3f}, pre: {:.3f}, f1: {:.3f}'.format(epoch, epoches, total_loss_epoch / total_tmp,
                                                                          auc, sen, pre, F1))

