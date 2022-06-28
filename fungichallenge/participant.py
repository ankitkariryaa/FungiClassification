import mysql.connector
import time

def connect():
    mydb = mysql.connector.connect(
        host="fungi.compute.dtu.dk",
        user="fungiuser",
        password="fungi_4Fun",
        database="fungi"
    )
    return mydb


def check_name_and_pw(team, team_pw):
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


def get_current_credits(team, team_pw):
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
    #n_request = n_request - inital_n_train

    sql = "SELECT credits FROM teams where name = %s"
    val = (team,)
    mycursor.execute(sql, val)
    myresults = mycursor.fetchall()
    total_credits = myresults[0][0]

    return total_credits - n_request


def requested_data(team, team_pw):
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
    for id in myresults:
        image_id = id[0]
        taxon_id = id[1]
        imgs_and_labels.append([image_id, taxon_id])

    return imgs_and_labels


def get_data_set(team, team_pw, dataset):
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


def request_labels(team, team_pw, image_ids):
    if not check_name_and_pw(team, team_pw):
        return 0

    current_credits = get_current_credits(team, team_pw)
    if len(image_ids) > current_credits:
        print("You have requested more ids than you have available credits")
        return None

    mydb = connect()
    mycursor = mydb.cursor()

    imgs_and_labels = []
    for im in image_ids:
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


def get_all_label_ids(team, team_pw):
    if not check_name_and_pw(team, team_pw):
        return None

    label_species = []
    mydb = connect()
    mycursor = mydb.cursor()
    sql = "SELECT taxonID, species_name FROM taxon_id_species"
    mycursor.execute(sql)
    myresults = mycursor.fetchall()
    for id in myresults:
        # print(id)
        taxonID = id[0]
        spec_name = id[1]
        label_species.append([taxonID, spec_name])
    return label_species


def submit_labels(team, team_pw, image_and_labels):
    if not check_name_and_pw(team, team_pw):
        return 0

    mydb = connect()
    mycursor = mydb.cursor()

    for sub in image_and_labels:
        img_id = sub[0]
        label = sub[1]
        time_now = time.strftime('%Y-%m-%d %H:%M:%S')
        sql = "INSERT INTO submitted_labels (image_id, team_name, label, submission_time) VALUES (%s, %s, %s, %s)"
        val = (img_id, team, label, time_now)
        mycursor.execute(sql, val)
        mydb.commit()
    print('Team', team, 'submitted', len(image_and_labels), 'labels')
