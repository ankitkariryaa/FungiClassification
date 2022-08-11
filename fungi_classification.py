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
from albumentations import Compose, Normalize, Resize
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
from efficientnet_pytorch import EfficientNet
import numpy as np
import tqdm
from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
import time


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

    labels = fcp.request_labels(tm, tm_pw, req_imgs)


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
    def __init__(self, df, transform=None):
        self.df = df
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
        return Compose([
            RandomResizedCrop(width, height, scale=(0.8, 1.0)),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(p=0.2),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return Compose([
            Resize(width, height),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    accumulation_steps = 2
    n_epochs = 20
    n_workers = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True, num_workers=n_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_sz, shuffle=False, num_workers=n_workers)

    seed_torch(777)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, n_classes)

    model.to(device)

    lr = 0.01
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)

    criterion = nn.CrossEntropyLoss()
    best_score = 0.
    best_loss = np.inf

    for epoch in range(n_epochs):
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


if __name__ == '__main__':
    # Your team and team password
    # team = "DancingDeer"
    # team_pw = "fungi44"
    team = "BigAnt"
    team_pw = "fungi66"

    # where is the full set of images placed
    image_dir = "C:/data/Danish Fungi/DF20M/"

    # where should log files, temporary files and trained models be placed
    network_dir = "C:/data/Danish Fungi/FungiNetwork/"

    get_participant_credits(team, team_pw)
    print_data_set_numbers(team, team_pw)
    # request_random_labels(team, team_pw)
    get_all_data_with_labels(team, team_pw, image_dir, network_dir)
    train_fungi_network(network_dir)
    evaluate_network_on_test_set(team, team_pw, image_dir, network_dir)
    compute_challenge_score(team, team_pw, network_dir)

