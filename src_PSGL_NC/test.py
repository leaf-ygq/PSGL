from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter

from sklearn.preprocessing import LabelBinarizer

from args import *
from model import *
from utils import *
from dataset import *

import sys
import numpy

def load_checkpoint(filepath,device):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    model.to(device)
    model.eval()
    return model



def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

numpy.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(edgeitems=sys.maxsize)


if not os.path.isdir('results'):
    os.mkdir('results')
if not os.path.isdir('epochs'):
    os.mkdir('epochs')
    
# args
args = make_args()
print(args)
np.random.seed(123)
np.random.seed()

# set up gpu
if args.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
else:
    print('Using CPU')
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

# args.cache = True

# for task in ['link', 'link_pair']:
for task in ['link_pair']:
# for task in ['link']:
    args.task = task
    if args.dataset=='All':
        if task == 'link':
            # datasets_name =['grid', 'communities']
            # datasets_name = ['grid']
            # datasets_name = ['communities']
            # datasets_name = ['grid']
            # datasets_name = ['ppi']
            datasets_name = []
        else:
            # datasets_name =['communities']
            # datasets_name = ['email']
            # datasets_name =['protein']
            # datasets_name = ['Cora']
            datasets_name =['Cora','CiteSeer']
    else:
        datasets_name = [args.dataset]
    for dataset_name in datasets_name:
        if not os.path.isdir(dataset_name):
            os.mkdir(dataset_name)

        timestart = time.time()
        
        results = []
        results2 = []
        graph_load_time=[]
        maxEpochs=[]
        maxTrainTimes=[]
        maxInfTimes=[]


        if args.weightedRandomWalk:
            rwstr='WeightedRandomWalk'
        else:
            rwstr='RandomWalk'
        if args.normalized:
            rwnormstr='Normalized'
        else:
            rwnormstr='UnNormalized'
        if args.edgelabel:
            eldl='LabeledEdge'
        else:
            eldl=''
        if args.attention:
            attn='withAttention'
        else:
            attn=''
        if args.fastRandomWalk:
            fastRW='Fast'
        else:
            fastRW=''
        if args.attentionAddSelf:
            addself_atten='attentionAddSelf'
        else:
            addself_atten=''



        for repeat in range(args.repeat_num):
            result_val = []
            result_test = []

            result_test2 = []
            max_Epoch=[]
            Train_Time=[]
            Inf_Time=[]
            maxTrainTime=[]
            maxInfTime=[]
            Embedding=[]
            anchorset_ids=[]
            time1 = time.time()
            data_list,_,edge_labels,data_dists_list = get_tg_dataset(args, dataset_name, use_cache=args.cache, remove_feature=args.rm_feature)
            time2 = time.time()
            load_time=time2-time1
            graph_load_time.append(load_time)

            num_features = data_list[0].x.shape[1]
            num_node_classes = None
            num_graph_classes = None
            if 'y' in data_list[0].__dict__ and data_list[0].y is not None:
                num_node_classes = max([data.y.max().item() for data in data_list])+1
            if 'y_graph' in data_list[0].__dict__ and data_list[0].y_graph is not None:
                num_graph_classes = max([data.y_graph.numpy()[0] for data in data_list])+1
            print('Dataset', dataset_name, 'Graph', len(data_list), 'Feature', num_features, 'Node Class', num_node_classes, 'Graph Class', num_graph_classes)
            nodes = [data.num_nodes for data in data_list]
            edges = [data.num_edges for data in data_list]
            print('Node: max{}, min{}, mean{}'.format(max(nodes), min(nodes), sum(nodes)/len(nodes)))
            print('Edge: max{}, min{}, mean{}'.format(max(edges), min(edges), sum(edges)/len(edges)))

            args.batch_size = min(args.batch_size, len(data_list))
            print('Anchor num {}, Batch size {}'.format(args.anchor_num, args.batch_size))

            
            for i,data in enumerate(data_list):

                # anchorset_ids=preselect_anchor(data,bipartite_list[i],data_dists_list[i],args,select_anchors=args.select_anchors, layer_num=args.layer_num, anchor_num=args.anchor_num, device='cpu')
                
                print('length:', data.x.shape[0])
                m = int(np.log2(data.x.shape[0]))
                args.anchor_num = m * m 
                # anchorset_ids = preselect_anchor2(data, data_dists_list[i], layer_num=args.layer_num, anchor_num=args.anchor_num, device=device)
                
                data = data.to(device)
                data_list[i] = data

            # model
            input_dim = num_features
            output_dim = args.output_dim

            if 'link' in args.task:
                out_act = nn.Sigmoid()


            model = load_checkpoint('{}_{}_{}_layer{}_{}Anchors_{}_{}{}{}_WalkLength{}_NumWalks{}_p{}_q{}{}_{}_Anchors_{}.pth'.format(args.task,args.model,dataset_name,args.layer_num,args.select_anchors,eldl,fastRW,rwnormstr,rwstr,args.walk_length,args.num_walks,args.p,args.q,attn,addself_atten,args.Num_Anchors),device)


            loss_train = 0
            loss_val = 0
            loss_test = 0
            correct_train = 0
            all_train = 0
            correct_val = 0
            all_val = 0
            correct_test = 0
            all_test = 0
            auc_train = 0
            auc_val = 0
            auc_test = 0
            emb_norm_min = 0
            emb_norm_max = 0
            emb_norm_mean = 0

            loss_test2 = 0
            correct_test2 = 0

            Inf_time_start = time.time()
            
            for id, data in enumerate(data_list):
                out = model(data, data_dists_list[id])
                emb_norm_min += torch.norm(out.data, dim=1).min().cpu().numpy()
                emb_norm_max += torch.norm(out.data, dim=1).max().cpu().numpy()
                emb_norm_mean += torch.norm(out.data, dim=1).mean().cpu().numpy()


                pred = model(data, data_dists_list[id])[data.Node_Classification_Train].max(1)[1]
                loss_train += F.nll_loss(model(data, data_dists_list[id])[data.Node_Classification_Train], data.y[data.Node_Classification_Train])
                auc_train += multiclass_roc_auc_score(data.y[data.Node_Classification_Train].flatten().cpu().numpy()   , pred.cpu().numpy().flatten())
                


                pred = model(data, data_dists_list[id])[data.Node_Classification_Val].max(1)[1]
                loss_val += F.nll_loss(model(data, data_dists_list[id])[data.Node_Classification_Val], data.y[data.Node_Classification_Val])
                auc_val += multiclass_roc_auc_score(data.y[data.Node_Classification_Val].flatten().cpu().numpy()   , pred.cpu().numpy().flatten())
                
                pred = model(data, data_dists_list[id])[data.Node_Classification_Test].max(1)[1]
                loss_test += F.nll_loss(model(data, data_dists_list[id])[data.Node_Classification_Test], data.y[data.Node_Classification_Test])
                auc_test += multiclass_roc_auc_score(data.y[data.Node_Classification_Test].flatten().cpu().numpy()   , pred.cpu().numpy().flatten())
                
            # for name, parms in model.named_parameters():	
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:',parms.requires_grad)
            #     print('-->grad_value:',parms.grad)
            #     print("===")

            loss_train /= id+1
            loss_val /= id+1
            loss_test /= id+1
            emb_norm_min /= id+1
            emb_norm_max /= id+1
            emb_norm_mean /= id+1
            auc_train /= id+1
            auc_val /= id+1
            auc_test /= id+1


            Inf_time_end = time.time()
            Inf_time=Inf_time_end-Inf_time_start  

            print(repeat, 'Test AUC: {:.4f}'.format(auc_test) )

        with open('results/InferenceTime_{}_{}_{}_layer{}_{}Anchors_{}_{}{}{}_WalkLength{}_NumWalks{}_p{}_q{}{}_{}_Anchors_{}.txt'.format(args.task,args.model,dataset_name,args.layer_num,args.select_anchors,eldl,fastRW,rwnormstr,rwstr,args.walk_length,args.num_walks,args.p,args.q,attn,addself_atten,args.Num_Anchors), 'w') as f:
            f.write('{}\n'.format(Inf_time))
