import fungichallenge.participant as fcp
import random

def test_get_participant_credits():
    team = "DancingDeer"
    team_pw = "fungi44"
    current_credits = fcp.get_current_credits(team, team_pw)
    print('Team', team, 'credits:', current_credits)


def test_get_data_set():
    team = "DancingDeer"
    team_pw = "fungi44"
    imgs_and_data = fcp.get_data_set(team, team_pw, 'train_set')
    print('train_set pairs', len(imgs_and_data))
    imgs_and_data = fcp.get_data_set(team, team_pw, 'train_labels_set')
    print('train_labels_set set pairs', len(imgs_and_data))
    imgs_and_data = fcp.get_data_set(team, team_pw, 'test_set')
    print('test_set set pairs', len(imgs_and_data))
    imgs_and_data = fcp.get_data_set(team, team_pw, 'final_set')
    print('final_set set pairs', len(imgs_and_data))
    imgs_and_data = fcp.get_data_set(team, team_pw, 'requested_set')
    print('requested_set set pairs', len(imgs_and_data))


def test_request_labels():
    team = "DancingDeer"
    team_pw = "fungi44"

    imgs_and_data = fcp.get_data_set(team, team_pw, 'train_set')
    n_img = len(imgs_and_data)

    req_imgs = []
    for i in range(10):
        idx = random.randint(0, n_img - 1)
        im_id = imgs_and_data[idx][0]
        req_imgs.append(im_id)

    # imgs = ['noimage', 'imge21']
    labels = fcp.request_labels(team, team_pw, req_imgs)
    print(labels)


def test_submit_labels():
    # team = "DancingDeer"
    # team_pw = "fungi44"
    team = "BigAnt"
    team_pw = "fungi66"

    imgs_and_data = fcp.get_data_set(team, team_pw, 'test_set')
    # n_img = len(imgs_and_data)

    label_and_species = fcp.get_all_label_ids(team, team_pw)
    n_label = len(label_and_species)

    im_and_labels = []
    for im in imgs_and_data:
        if random.randint(0, 100) > 70:
            im_id = im[0]
            rand_label_idx = random.randint(0, n_label - 1)
            rand_label = label_and_species[rand_label_idx][0]
            im_and_labels.append([im_id, rand_label])

    fcp.submit_labels(team, team_pw, im_and_labels)


def test_compute_score():
    # team = "DancingDeer"
    # team_pw = "fungi44"
    team = "BigAnt"
    team_pw = "fungi66"

    results = fcp.compute_score(team, team_pw)
    print(results)


if __name__ == '__main__':
    # test_get_participant_credits()
    # test_get_data_set()
    # test_request_labels()
    # test_submit_labels()
    test_compute_score()
