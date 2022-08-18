import os.path
import sys
import pandas as pd
import fungichallenge.participant as fcp
import random
import torch
import torch.nn as nn
import cv2
from torch.optim import Adam, SGD, AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
from efficientnet_pytorch import EfficientNet
import numpy as np
import tqdm
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
import time


class EfficientNetWithFeatures(EfficientNet):
    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        ef = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(ef)
        x = x.view(bs, -1)
        x = self._dropout(x)
        r = self._fc(x)
        return r, ef

def get_participant_credits(tm, tm_pw):
    """
        Print available credits for the team
    """
    current_credits = fcp.get_current_credits(tm, tm_pw)
    print('Team', team, 'credits:', current_credits)


def print_data_set_numbers(tm, tm_pw):
    """
    Debug test function to get data
    """
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'train_set')
    print('train_set pairs', len(imgs_and_data))
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'train_labels_set')
    print('train_labels_set set pairs', len(imgs_and_data))
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'test_set')
    print('test_set set pairs', len(imgs_and_data))
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'final_set')
    print('final_set set pairs', len(imgs_and_data))
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'requested_set')
    print('requested_set set pairs', len(imgs_and_data))


def request_random_labels(tm, tm_pw):
    """
    An example on how to request labels from the available pool of images.
    Here it is just a random subset being requested
    """
    n_request = 500

    # First get the image ids from the pool
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'train_set')
    n_img = len(imgs_and_data)
    print("Number of images in training pool (no labels)", n_img)

    req_imgs = []
    for i in range(n_request):
        idx = random.randint(0, n_img - 1)
        im_id = imgs_and_data[idx][0]
        req_imgs.append(im_id)

    # labels = fcp.request_labels(tm, tm_pw, req_imgs)


def test_submit_labels(tm, tm_pw):
    """
        Submitting random labels for testing
    """
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'test_set')
    label_and_species = fcp.get_all_label_ids(tm, team_pw)
    n_label = len(label_and_species)

    im_and_labels = []
    for im in imgs_and_data:
        if random.randint(0, 100) > 70:
            im_id = im[0]
            rand_label_idx = random.randint(0, n_label - 1)
            rand_label = label_and_species[rand_label_idx][0]
            im_and_labels.append([im_id, rand_label])

    fcp.submit_labels(tm, tm_pw, im_and_labels)


def get_all_data_with_labels(tm, tm_pw, id_dir, nw_dir):
    """
        Get the team data that has labels (initial data plus requested data).
        Writes a csv file with the image names and their class ids.
        Also writes a csv file with some useful statistics
    """
    stats_out = os.path.join(nw_dir, "fungi_class_stats.csv")
    data_out = os.path.join(nw_dir, "data_with_labels.csv")

    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'train_labels_set')
    imgs_and_data_r = fcp.get_data_set(tm, tm_pw, 'requested_set')
    print("Team", tm, ' has access to images with labels:\n',
          'Basis set:', len(imgs_and_data), '\n',
          'Requested set:', len(imgs_and_data_r))

    total_img_data = imgs_and_data + imgs_and_data_r
    df = pd.DataFrame(total_img_data, columns=['image', 'taxonID'])
    # print(df.head())
    all_taxon_ids = df['taxonID']

    # convert taxonID into a class id
    taxon_id_to_label = {}
    # label_to_taxon_id = {}
    for count, value in enumerate(all_taxon_ids.unique()):
        taxon_id_to_label[int(value)] = count
        # label_to_taxon_id[count] = int(value)

    with open(data_out, 'w') as f:
        f.write('image,class\n')
        for t in total_img_data:
            class_id = taxon_id_to_label[t[1]]
            out_str = os.path.join(id_dir, t[0]) + '.JPG, ' + str(class_id) + '\n'
            f.write(out_str)

    with open(stats_out, 'w') as f:
        f.write('taxonID,class,count\n')
        for ti in taxon_id_to_label:
            count = df['taxonID'].value_counts()[ti]
            class_id = taxon_id_to_label[ti]
            out_str = str(ti) + ', ' + str(class_id) + ', ' + str(count) + '\n'
            f.write(out_str)


class NetworkFungiDataset(Dataset):
    def __init__(self, df, transform=None, assign_labels=False):
        self.df = df
        if assign_labels:
            self.df['labels'] = df['class'].map(lambda x: int(x) if type(x) is not None else 0)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['image'].values[idx]
        label = 0
        if self.df['class'].values[idx] is not None:
            label = int(self.df['class'].values[idx])
        try:
            image = cv2.imread(file_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Fails on GPU Cluster - never stops
        except cv2.error as e:
            print("OpenCV error with", file_path, "error", e)
        except IOError:
            print("IOError with", file_path)
        except:
            print("Could not read or convert", file_path)
            print(sys.exc_info())
            return None, None

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label


def get_transforms(data):
    
    width = 299
    height = 299

    if data == 'train':
        
        transform = A.Compose([
            #A.CenterCrop(p=0.5, width=width, height=height),
            A.RandomResizedCrop(p=1, width=width, height=height, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(p=0.5, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.RandomBrightnessContrast(p=0.25),
            A.HueSaturationValue(p=0.25),
            A.GaussNoise(var_limit=(0.5, 2), p=0.2),
            A.OneOf([
                #A.MotionBlur(blur_limit=3),
                A.MedianBlur(blur_limit=3),
                #A.Blur(blur_limit=3),
                A.OneOf([
                    A.Sharpen(),
                    A.Emboss(),
                        ],
                p=0.1)]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        
        return transform
    
    elif data == 'valid':
        return A.Compose([
            A.Resize(width, height),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        print("Unknown data set requested")
        return None


def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_logger(log_file='train.log'):
    log_format = '%(asctime)s %(levelname)s %(message)s'

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger

def get_sample_batch(labels, dataset, postive=True):
    px = []
    for i in labels:
        if postive:
            l = dataset.df.query(f"labels == {i}")
        else:
            l = dataset.df.query(f"labels != {i}")
        if len(l) == 0:
            return None
        else:
            j = random.choice(l.index.tolist())
            px.append(dataset[j][0]) # only the image
    return torch.stack(px, dim=0)


def pretrain_fungi_network(nw_dir):
    data_file = os.path.join(nw_dir, "data_with_labels.csv")
    log_file = os.path.join(nw_dir, "FungiEfficientNet-B0.log")
    logger = init_logger(log_file)

    df = pd.read_csv(data_file)
    n_classes = len(df['class'].unique())
    print("Number of classes in data", n_classes)
    print("Number of samples with labels", df.shape[0])

    train_dataset = NetworkFungiDataset(df, transform=get_transforms(data='train'), assign_labels=True)
    # TODO: Divide data into training and validation
    valid_dataset = NetworkFungiDataset(df, transform=get_transforms(data='valid'), assign_labels=True)

    # batch_sz * accumulation_step = 64
    batch_sz = 12
    accumulation_steps = 6
    n_epochs = 50
    n_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=n_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_sz, shuffle=False, num_workers=n_workers)

    seed_torch(777)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = EfficientNetWithFeatures.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, n_classes)
    model.to(device)

    lr = 0.01
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)

    margin_loss = nn.TripletMarginLoss()
    best_score = 0.
    best_loss = np.inf

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        optimizer.zero_grad()

        print("Pre Training")

        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):
            images = images.to(device)

            positive_samples = get_sample_batch(labels, train_dataset, postive=True)
            negative_samples = get_sample_batch(labels, train_dataset, postive=False)
            if positive_samples is None or negative_samples is None:
                continue
            positive_samples = positive_samples.to(device)
            negative_samples = negative_samples.to(device)

            _, anchor_feature_preds = model(images)
            _, positive_feature_preds = model(positive_samples)
            _, negative_feature_preds = model(negative_samples)

            loss = margin_loss(anchor_feature_preds, positive_feature_preds, negative_feature_preds)

            # Scale the loss to the mean of the accumulated batch size
            loss = loss / accumulation_steps
            loss.backward()
            if (i - 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += loss.item() / len(train_loader)

        print("Doing validation for pre training")
        model.eval()
        avg_val_loss = 0.

        for i, (images, labels) in tqdm.tqdm(enumerate(valid_loader)):
            images = images.to(device)
            labels = labels.to(device)
            positive_samples = get_sample_batch(labels, train_dataset, postive=True)
            negative_samples = get_sample_batch(labels, train_dataset, postive=False)
            if positive_samples is None or negative_samples is None:
                continue
            positive_samples = positive_samples.to(device)
            negative_samples = negative_samples.to(device)

            with torch.no_grad():

                _, anchor_feature_preds = model(images)
                _, positive_feature_preds = model(positive_samples)
                _, negative_feature_preds = model(negative_samples)


            loss = margin_loss(anchor_feature_preds, positive_feature_preds, negative_feature_preds)
            avg_val_loss += loss.item() / len(valid_loader)


        scheduler.step(avg_val_loss)

        elapsed = time.time() - start_time

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            logger.debug(f'  Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            best_model_name = os.path.join(nw_dir, "DF20M-EfficientNet-B0_best_loss_pretrained.pth")
            torch.save(model.state_dict(), best_model_name)


def train_fungi_network(nw_dir):
    data_file = os.path.join(nw_dir, "data_with_labels.csv")
    log_file = os.path.join(nw_dir, "FungiEfficientNet-B0.log")
    logger = init_logger(log_file)

    df = pd.read_csv(data_file)
    n_classes = len(df['class'].unique())
    print("Number of classes in data", n_classes)
    print("Number of samples with labels", df.shape[0])

    train_dataset = NetworkFungiDataset(df, transform=get_transforms(data='train'))
    # TODO: Divide data into training and validation
    valid_dataset = NetworkFungiDataset(df, transform=get_transforms(data='valid'))

    # batch_sz * accumulation_step = 64
    batch_sz = 32
    accumulation_steps = 6
    n_epochs = 50
    n_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=n_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_sz, shuffle=False, num_workers=n_workers)

    seed_torch(777)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    pretrained_model_path = os.path.join(nw_dir, "DF20M-EfficientNet-B0_best_loss_pretrained.pth")

    if os.path.exists(pretrained_model_path):
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model.load_state_dict(torch.load(pretrained_model_path))
        model.to(device)
    else:
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, n_classes)
        model.to(device)

    lr = 0.01
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)
    data_stats_file = os.path.join(nw_dir, "fungi_class_stats.csv")
    data_stats = pd.read_csv(data_stats_file)
    cls_num_list = data_stats['count']
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, cls_num_list)
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
    per_cls_weights = torch.FloatTensor(per_cls_weights).to(device=device)
    print(f"Class weights: {per_cls_weights}")

    criterion = nn.CrossEntropyLoss()
    best_score = 0.
    best_loss = np.inf

    for epoch in range(n_epochs):
        if epoch/n_epochs > 0.8:
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights)
        start_time = time.time()
        model.train()
        avg_loss = 0.
        optimizer.zero_grad()

        print("Training")

        for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            y_preds = model(images)
            loss = criterion(y_preds, labels)

            # Scale the loss to the mean of the accumulated batch size
            loss = loss / accumulation_steps
            loss.backward()
            if (i - 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                avg_loss += loss.item() / len(train_loader)

        print("Doing validation")
        model.eval()
        avg_val_loss = 0.
        preds = np.zeros((len(valid_dataset)))
        preds_raw = []

        for i, (images, labels) in tqdm.tqdm(enumerate(valid_loader)):
            images = images.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                y_preds = model(images)

            preds[i * batch_sz: (i + 1) * batch_sz] = y_preds.argmax(1).to('cpu').numpy()
            preds_raw.extend(y_preds.to('cpu').numpy())

            loss = criterion(y_preds, labels)
            avg_val_loss += loss.item() / len(valid_loader)

        scheduler.step(avg_val_loss)

        # TODO: Divide data into training and validation
        score = f1_score(df['class'], preds, average='macro')
        accuracy = accuracy_score(df['class'], preds)
        recall_3 = top_k_accuracy_score(df['class'], preds_raw, k=3)

        elapsed = time.time() - start_time
        logger.debug(
          f'  Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f} F1: {score:.6f}  Accuracy: {accuracy:.6f} Recall@3: {recall_3:.6f} time: {elapsed:.0f}s')

        if accuracy > best_score:
            best_score = accuracy
            logger.debug(f'  Epoch {epoch + 1} - Save Best Accuracy: {best_score:.6f} Model')
            best_model_name = os.path.join(nw_dir, "DF20M-EfficientNet-B0_best_accuracy.pth")
            torch.save(model.state_dict(), best_model_name)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            logger.debug(f'  Epoch {epoch + 1} - Save Best Loss: {best_loss:.4f} Model')
            best_model_name = os.path.join(nw_dir, "DF20M-EfficientNet-B0_best_loss.pth")
            torch.save(model.state_dict(), best_model_name)


def evaluate_network_on_test_set(tm, tm_pw, im_dir, nw_dir):
    """
        Evaluate trained network on the test set and submit the results to the challenge database.
        The scores can be extracted using compute_challenge_score.
        The function can also be used to evaluate on the final set
    """
    # Use 'test-set' for the set of data that can evaluated several times
    # Use 'final-set' for the final set that will be used in the final score of the challenge
    use_set = 'test_set'
    # use_set = 'final_set'
    print(f"Evaluating on {use_set}")

    best_trained_model = os.path.join(nw_dir, "DF20M-EfficientNet-B0_best_accuracy.pth")
    log_file = os.path.join(nw_dir, "FungiEvaluation.log")
    data_stats_file = os.path.join(nw_dir, "fungi_class_stats.csv")

    # Debug on model trained elsewhere
    # best_trained_model = os.path.join("C:/data/Danish Fungi/training/", "DF20M-EfficientNet-B0_best_accuracy - Copy.pth")
    # data_stats_file = os.path.join("C:/data/Danish Fungi/training/", "class-stats.csv")

    logger = init_logger(log_file)

    imgs_and_data = fcp.get_data_set(team, team_pw, use_set)
    df = pd.DataFrame(imgs_and_data, columns=['image', 'class'])
    df['image'] = df.apply(
        lambda x: im_dir + x['image'] + '.JPG', axis=1)

    test_dataset = NetworkFungiDataset(df, transform=get_transforms(data='valid'))

    batch_sz = 32
    n_workers = 8
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False, num_workers=n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    n_classes = 183
    model = EfficientNet.from_name('efficientnet-b0', num_classes=n_classes)
    checkpoint = torch.load(best_trained_model)
    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()
    preds = np.zeros((len(test_dataset)))
    # preds_raw = []

    for i, (images, labels) in tqdm.tqdm(enumerate(test_loader)):
        images = images.to(device)

        with torch.no_grad():
            y_preds = model(images)

        preds[i * batch_sz: (i + 1) * batch_sz] = y_preds.argmax(1).to('cpu').numpy()
        # preds_raw.extend(y_preds.to('cpu').numpy())

    # Transform classes into taxonIDs
    data_stats = pd.read_csv(data_stats_file)
    img_and_labels = []
    for i, s in enumerate(imgs_and_data):
        pred_class = int(preds[i])
        taxon_id = int(data_stats['taxonID'][data_stats['class'] == pred_class])
        img_and_labels.append([s[0], taxon_id])

    print("Submitting labels")
    fcp.submit_labels(tm, tm_pw, img_and_labels)


def compute_challenge_score(tm, tm_pw, nw_dir):
    """
        Compute the scores on the test set using the result submitted to the challenge database.
    """
    log_file = os.path.join(nw_dir, "FungiScores.log")
    logger = init_logger(log_file)
    results = fcp.compute_score(tm, tm_pw)
    # print(results)
    logger.info(results)


def forward_pass_no_labels(tm, tm_pw, im_dir, nw_dir):
    """
    Function to do forward pass on unlabbelled training data
    
    Outputs softmax predictions for all images as rows with the coorespopnding image list
s
    """
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'train_set')
    df = pd.DataFrame(imgs_and_data, columns=['image', 'class'])
    df['image'] = df.apply(
        lambda x: im_dir + x['image'] + '.JPG', axis=1)
    
    best_trained_model = os.path.join(nw_dir, "DF20M-EfficientNet-B0_best_accuracy.pth")

    # Debug on model trained elsewhere
    # best_trained_model = os.path.join("C:/data/Danish Fungi/training/", "DF20M-EfficientNet-B0_best_accuracy - Copy.pth")
    # data_stats_file = os.path.join("C:/data/Danish Fungi/training/", "class-stats.csv")

    unlabbel_dataset = NetworkFungiDataset(df, transform=get_transforms(data='valid'))

    batch_sz = 32
    n_workers = 4
    
    unlabbel_loader = DataLoader(unlabbel_dataset, batch_size=batch_sz, shuffle=False, num_workers=n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    n_classes = 183
    model = EfficientNetWithFeatures.from_pretrained('efficientnet-b0', num_classes=n_classes)
    checkpoint = torch.load(best_trained_model)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    _avg_pool = nn.AdaptiveAvgPool2d(1)
    preds = np.zeros((len(imgs_and_data), n_classes))
    features = np.zeros((len(imgs_and_data), model._fc.in_features))
    
    for i, (images, labels) in tqdm.tqdm(enumerate(unlabbel_loader)):
        images = images.to(device)

        with torch.no_grad():
            y_preds, feats = model(images)

        preds[i*batch_sz : (i+1)*batch_sz,:] = y_preds.softmax(dim=1).to('cpu').numpy()
        features[i*batch_sz : (i+1)*batch_sz,:] =  _avg_pool(feats).squeeze().squeeze().to('cpu').numpy()
        
    np.savez(os.path.join(nw_dir,"features_and_softmax.npz"),softmax_scores=preds, features=features)
    # return preds
        
def request_labels(tm, tm_pw, im_dir, nw_dir):
    """
    An example on how to request labels from the available pool of images.
    Here it is just a random subset being requested
    """
    n_request = 500

    # First get the image ids from the pool
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'train_set')
    
    n_img = len(imgs_and_data)
    print("Number of images in training pool (no labels)", n_img)
    
    softmax_scores = forward_pass_no_labels(tm, tm_pw, im_dir, nw_dir)

    #TODO
    # use softmax_scores for entropy sampling here

    req_imgs = []
    for i in range(n_request):
        idx = random.randint(0, n_img - 1)
        im_id = imgs_and_data[idx][0]
        req_imgs.append(im_id)

    # labels = fcp.request_labels(tm, tm_pw, req_imgs)

def request_specific_labels(tm, tm_pw):
    # First get the image ids from the pool
    imgs_and_data = fcp.get_data_set(tm, tm_pw, 'train_set')
    image_idx = [t[0] for t in imgs_and_data]
    softmax_scores = np.load(os.path.join(network_dir,"softmax_scores.npy"))
    
    # idxs_kmeans = kmeans_sample(softmax_scores, n_sample=200)
    idxs_entropy = entropy_sample(softmax_scores, n_sample=1999)
    
    image_entropy = np.array(image_idx)[idxs_entropy].tolist()
    
    answer = input("Are you sure you want to spent credits[y/n]: ")
    
    if answer.lower() == "y":
        fcp.request_labels(tm, tm_pw, image_entropy)
    
    return

    # labels = fcp.request_labels(tm, tm_pw, im_ids)
    # return labels

def entropy_sample(probs, n_sample):
    log_probs = np.log(probs)
    U = (probs*log_probs).sum(1)
    idxs = np.argsort(U)[:n_sample]
    return idxs

def kmeans_sample(embedding, n_sample):
    from sklearn.cluster import KMeans
    
    cluster_learner = KMeans(n_clusters=n_sample)
    cluster_learner.fit(embedding)
  		
    cluster_idxs = cluster_learner.predict(embedding)
    centers = cluster_learner.cluster_centers_[cluster_idxs]
    dis = (embedding - centers)**2
    dis = dis.sum(axis=1)
    q_idxs = np.array([np.arange(embedding.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n_sample)])
    
    return q_idxs

if __name__ == '__main__':
    # Your team and team password
    # team = "DancingDeer"
    # team_pw = "fungi44"
    team = "FunnyFly"
    team_pw = "fungi89"

    # where is the full set of images placed
    image_dir = "C:/Users/lowes/OneDrive/Skrivebord/DTU/8_semester/summerschool/data/DF20M/"

    # where should log files, temporary files and trained models be placed
    network_dir = "C:/Users/lowes/OneDrive/Skrivebord/DTU/8_semester/summerschool/src/FungiNet/"


    get_participant_credits(team, team_pw)
    
    # request_specific_labels(team, team_pw)
    
    print_data_set_numbers(team, team_pw)
    
    # forward_pass_no_labels(team, team_pw, image_dir, network_dir)
    



    pretrain_fungi_network(network_dir)

    train_fungi_network(network_dir)
    evaluate_network_on_test_set(team, team_pw, image_dir, network_dir)
    compute_challenge_score(team, team_pw, network_dir)

