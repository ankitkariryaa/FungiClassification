import fungichallenge.participant as fcp


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


if __name__ == '__main__':
    # test_get_participant_credits()
    test_get_data_set()
    