"""
    Contains functions to connect to the fungi challenge backend SQL server.
    It is possible to request data and submit results.
    """
import mysql.connector
import time
import sklearn.metrics
import datetime
from tqdm import tqdm


def connect():
    """
        Connect to the SQL backend server

        :param number: The number to multiply.
        :type number: int

        :param muiltiplier: The multiplier.
        :type muiltiplier: int

        :return: The connection class if connection was successful, `None` otherwise
        :rtype: MySQL connector
        """
    try:
        mydb = mysql.connector.connect(
            host="fungi.compute.dtu.dk",
            user="fungiuser",
            password="fungi_4Fun",
            database="fungi"
        )
        return mydb
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    return None


def check_name_and_pw(team, team_pw):
    """
        Verify the team name and team password in the backend server

        :param team: Name of the team
        :type team: string

        :param team_pw: Password.
        :type team_pw: string

        :return: `True` if team and password are correct, `False` otherwise
        :rtype: bool
        """
    try:
        mydb = connect()
        mycursor = mydb.cursor()
        sql = "SELECT password FROM teams where name = %s"
        val = (team,)
        mycursor.execute(sql, val)
        myresults = mycursor.fetchall()
        n_entries = len(myresults)
        if n_entries < 1:
            print('Team not found:', team)
            return False
        pw = myresults[0][0]
        if pw != team_pw:
            print('Team name and password does not match')
            return False
        return True
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    return False


def get_current_credits(team, team_pw):
    """
        Get the amount of credits that the team has.

        :param team: Name of the team
        :type team: string

        :param team_pw: Password.
        :type team_pw: string

        :return: Amount of credits
        :rtype: int
        """
    try:
        if not check_name_and_pw(team, team_pw):
            return 0

        mydb = connect()
        mycursor = mydb.cursor()
        # The user can have asked several times, we only want to count on (image_id, team) once
        sql = "SELECT COUNT(DISTINCT image_id, team_name) FROM requested_image_labels where team_name = %s"
        val = (team,)
        mycursor.execute(sql, val)
        myresults = mycursor.fetchall()
        n_request = myresults[0][0]

        sql = "SELECT credits FROM teams where name = %s"
        val = (team,)
        mycursor.execute(sql, val)
        myresults = mycursor.fetchall()
        total_credits = myresults[0][0]

        return total_credits - n_request
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    return None


def requested_data(team, team_pw):
    """
        Get the data with labels that have been requested/bought for credits

        :param team: Name of the team
        :type team: string

        :param team_pw: Password.
        :type team_pw: string

        :return: list of pairs of [image id, image label]
        :rtype: list of pairs
        """
    try:
        if not check_name_and_pw(team, team_pw):
            return 0

        mydb = connect()
        mycursor = mydb.cursor()
        sql = "select t1.image_id, t2.taxonID from requested_image_labels as t1 inner join " \
              "fungi_data as t2 on t1.image_id = t2.image_id where t1.team_name = %s"
        val = (team,)
        mycursor.execute(sql, val)
        myresults = mycursor.fetchall()

        imgs_and_labels = []
        for idx in myresults:
            image_id = idx[0]
            taxon_id = idx[1]
            imgs_and_labels.append([image_id, taxon_id])

        return imgs_and_labels
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    return None


def get_data_set(team, team_pw, dataset):
    """
        Get a given data set with or without labels.
        It returns a list of [image id, label] pairs, where label='None' if the label is not available.

        train_set : The set of data that can be used for training but without given labels.
                    It is possible to buy the labels from this set. If a label a bought, the
                    id is copied into the 'requested_set'

        train_labels_set : The set of data where the labels are given from the start.

        requested_set : The set of data, where a team has bought the labels using credits.

        test_set : The set that will be used for computing intermediate scores during the challenge.
                   Can be considered as a validation set, but where only organizers have the labels.

         final_set : The set that will be used for the final score.

        :param team: Name of the team
        :type team: string

        :param team_pw: Password.
        :type team_pw: string

        :param dataset: The wanted dataset.
        :type dataset: string

        :return: list of pairs of [image id, image label]
        :rtype: list of pairs
        """
    try:
        if not check_name_and_pw(team, team_pw):
            return 0

        available_set = ['train_set', 'train_labels_set', 'test_set', 'final_set', 'requested_set']
        if dataset not in available_set:
            print('Requested data set', dataset, 'not in:', available_set)
            return None

        mydb = connect()
        mycursor = mydb.cursor()

        imgs_and_labels = []
        if dataset == 'train_labels_set':
            sql = "select image_id, taxonID from fungi_data where dataset = %s"
            val = (dataset,)
            mycursor.execute(sql, val)
            myresults = mycursor.fetchall()
            for id in myresults:
                image_id = id[0]
                taxon_id = id[1]
                imgs_and_labels.append([image_id, taxon_id])
        elif dataset == 'requested_set':
            return requested_data(team, team_pw)
        else:
            sql = "select image_id, taxonID from fungi_data where dataset = %s"
            val = (dataset,)
            mycursor.execute(sql, val)
            myresults = mycursor.fetchall()
            for id in myresults:
                image_id = id[0]
                taxon_id = None
                imgs_and_labels.append([image_id, taxon_id])

        return imgs_and_labels
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    return None


def request_labels(team, team_pw, image_ids):
    """
        Request the labels from a part of the 'training_set' image ids. It costs credits
        to request labels. It returns the [image id, labels] pairs of the requested image ids.
        The requested [image id, labels] pairs can later be retrieved using get_data_set.

        :param team: Name of the team
        :type team: string

        :param team_pw: Password.
        :type team_pw: string

        :param image_ids: List of image ids.
        :type team_pw: list

        :return: list of pairs of [image id, image label]
        :rtype: list of pairs
        """
    try:
        if not check_name_and_pw(team, team_pw):
            return 0

        current_credits = get_current_credits(team, team_pw)
        if len(image_ids) > current_credits:
            print("You have requested more ids than you have available credits")
            return None

        mydb = connect()
        mycursor = mydb.cursor()

        imgs_and_labels = []
        for i in tqdm(range(len(image_ids))):
            im = image_ids[i]
            dataset = 'train_set'
            sql = "SELECT taxonID from fungi_data where image_id = %s and dataset = %s"
            val = (im, dataset)
            mycursor.execute(sql, val)
            myresults = mycursor.fetchall()
            if len(myresults) > 0:
                imgs_and_labels.append([im, myresults[0][0]])
                time_now = time.strftime('%Y-%m-%d %H:%M:%S')
                sql = "INSERT INTO requested_image_labels (image_id, team_name, request_time) VALUES (%s, %s, %s)"
                val = (im, team, time_now)
                mycursor.execute(sql, val)
                mydb.commit()
            elif len(myresults) == 0:
                print('Image with id', im, 'is not in the available training set')
            elif len(myresults) > 1:
                print('More than one hit found for', im, '- weird!')

        return imgs_and_labels
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    return None


def get_all_label_ids(team, team_pw):
    """
        Get a list of pairs [label, species name], where the label is an integer
        and the species name is a string with the scientific name of the species.

        :param team: Name of the team
        :type team: string

        :param team_pw: Password.
        :type team_pw: string

        :return: list of pairs of [label, species name]
        :rtype: list of pairs
        """
    try:
        if not check_name_and_pw(team, team_pw):
            return None

        label_species = []
        mydb = connect()
        mycursor = mydb.cursor()
        sql = "SELECT taxonID, species_name FROM taxon_id_species"
        mycursor.execute(sql)
        myresults = mycursor.fetchall()
        for idx in myresults:
            # print(id)
            taxonID = idx[0]
            spec_name = idx[1]
            label_species.append([taxonID, spec_name])
        return label_species
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    return None


def submit_labels(team, team_pw, image_and_labels):
    """
        Submit classification results as a list of pairs [image id, label].
        The time of submissions is kept for each submission and when computing scores
        only the most recent submission is used.

        :param team: Name of the team
        :type team: string

        :param team_pw: Password.
        :type team_pw: string

        :param image_and_labels: list of pairs of [image id, label]
        :type image_and_labels: list of pairs
        """
    try:
        if not check_name_and_pw(team, team_pw):
            return 0

        print("Submitting labels")
        mydb = connect()
        mycursor = mydb.cursor()

        # https://dev.mysql.com/doc/connector-python/en/connector-python-api-mysqlcursor-executemany.html
        val_list = []
        for i in range(len(image_and_labels)):
            sub = image_and_labels[i]
            img_id = sub[0]
            label = sub[1]
            time_now = time.strftime('%Y-%m-%d %H:%M:%S')
            val_list.append([img_id, team, label, time_now])
        sql = "INSERT INTO submitted_labels (image_id, team_name, label, submission_time) VALUES (%s, %s, %s, %s)"
        val = (img_id, team, label, time_now)
        mycursor.executemany(sql, val_list)

        # for i in tqdm(range(len(image_and_labels))):
        #     sub = image_and_labels[i]
        #     img_id = sub[0]
        #     label = sub[1]
        #     time_now = time.strftime('%Y-%m-%d %H:%M:%S')
        #     sql = "INSERT INTO submitted_labels (image_id, team_name, label, submission_time) VALUES (%s, %s, %s, %s)"
        #     val = (img_id, team, label, time_now)
        #     mycursor.execute(sql, val)
        mydb.commit()
        print('Team', team, 'submitted', len(image_and_labels), 'labels')
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))


def compute_score(team, team_pw):
    """
        Compute the current score on the test.

        :param team: Name of the team
        :type team: string

        :param team_pw: Password.
        :type team_pw: string

        :return Dictionary with results in different metrics.
        :rtype Dictionary with results.
        """
    try:
        if not check_name_and_pw(team, team_pw):
            return 0
        print(team)

        mydb = connect()
        mycursor = mydb.cursor()

        dataset = 'test_set'
        # Get ground truth
        sql = "select image_id, taxonID from fungi_data where dataset=%s"
        val = (dataset,)
        mycursor.execute(sql, val)
        ground_truth = mycursor.fetchall()
        n_entries_gt = len(ground_truth)
        print('Got', n_entries_gt, ' ground truth labels for set:', dataset)

        # Get submissions from team
        sql = "select image_id, label, submission_time from submitted_labels where team_name=%s"
        val = (team,)
        mycursor.execute(sql, val)
        submissions = mycursor.fetchall()
        n_entries = len(submissions)
        print('Got', n_entries, ' submitted labels from team:', team)

        # Use a nested dictionary
        pred_gt_dict = {}
        # add ground truth
        for gt in ground_truth:
            im_id = gt[0]
            tax_id = gt[1]
            pred_gt_dict[im_id] = {"gt": tax_id, "pred": -1, "time": datetime.datetime(1900, 1, 1)}

        # now update with latest submissions
        latest_time = datetime.datetime(1900, 1, 1)
        for sub in submissions:
            im_id_sub = sub[0]
            tax_id_sub = sub[1]
            time_sub = sub[2]
            # We can have submissions from other sets
            if im_id_sub in pred_gt_dict:
                cur_gt = pred_gt_dict[im_id_sub]["gt"]
                cur_time = pred_gt_dict[im_id_sub]["time"]
                if time_sub > cur_time:
                    pred_gt_dict[im_id_sub]["pred"] = tax_id_sub
                    pred_gt_dict[im_id_sub]["time"] = time_sub
                if time_sub > latest_time:
                    latest_time = time_sub

        y_true = []
        y_pred = []
        n_preds = 0
        for pg in pred_gt_dict.values():
            y_true.append(pg["gt"])
            y_pred.append(pg["pred"])
            if pg["pred"] > -1:
                n_preds = n_preds + 1

        print('Found', n_preds, ' predictions out of', n_entries_gt, 'Latest submission on', latest_time)
        # We use all prediction, where non-submitted are set to -1
        accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
        f1_score = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
        cohen_kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
        print('Accuracy:', accuracy, 'f1', f1_score, 'Cohen kappa', cohen_kappa)

        results_metric = {
            "Accuracy": accuracy,
            "F1score": f1_score,
            "CohenKappa": cohen_kappa,
            "LastSubmissionTime": latest_time
        }

        return results_metric
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    return None
