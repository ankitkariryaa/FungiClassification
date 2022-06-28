# Data
(https://sites.google.com/view/danish-fungi-dataset)
(chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://openaccess.thecvf.com/content/WACV2022/papers/Picek_Danish_Fungi_2020_-_Not_Just_Another_Image_Recognition_Dataset_WACV_2022_paper.pdf)

# Similar challenges
(https://www.kaggle.com/c/fungi-challenge-fgvc-2018)
(https://github.com/visipedia/fgvcx_fungi_comp#data)

# Creating Python lib
(https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f)
(https://packaging.python.org/en/latest/tutorials/packaging-projects/)
(https://towardsdatascience.com/deep-dive-create-and-publish-your-first-python-library-f7f618719e14)

# Python mysql
(https://www.w3schools.com/python/python_mysql_getstarted.asp)

# SQL help
(https://dev.mysql.com/doc/mysql-getting-started/en/)
(https://teaching.healthtech.dtu.dk/22112/index.php/Databases)
(https://dev.mysql.com/doc/refman/8.0/en/creating-tables.html)


## SQL commands

CREATE DATABASE fungi;
USE fungi;


CREATE TABLE teams
(
  id              INT unsigned NOT NULL AUTO_INCREMENT, # Unique ID for the record
  name            VARCHAR(150) NOT NULL,                # Name of the team
  credits         INT NOT NULL,                         # Teams credit
  PRIMARY KEY     (id)                                  # Make the id the primary key
);

INSERT INTO teams (name, credits) VALUES
('TeamHappyFrog', 10000),
('TeamFunkyDeer', 10000),
('TeamBrightSlug', 10000);

SELECT * FROM teams;

SELECT credits FROM teams WHERE name = 'TeamFunkyDeer';

## Updating credits

UPDATE teams set credits = credits - 1 where name = 'TeamFunkyDeer';


## user control in SQL
(https://www.thegeekdiary.com/beginners-guide-to-mysql-user-management/)

CREATE USER 'fungiuser'@'%'
  IDENTIFIED BY 'fungi_4Fun';
GRANT SELECT, UPDATE
  ON fungi.teams
  TO 'fungiuser'@'%';
GRANT SELECT
  ON fungi.fungi_data
  TO 'fungiuser'@'%';
GRANT SELECT, UPDATE, INSERT
  ON fungi.requested_image_labels
  TO 'fungiuser'@'%';
GRANT SELECT
  ON fungi.taxon_id_species
  TO 'fungiuser'@'%';
GRANT UPDATE, INSERT
  ON fungi.submitted_labels
  TO 'fungiuser'@'%';



CREATE USER 'fungisuper'@'%.dk'
  IDENTIFIED BY 'fungi38PW_';
GRANT ALL
  ON fungi.*
  TO 'fungisuper'@'%.dk';

CREATE USER 'fungisuper'@'%.net'
  IDENTIFIED BY 'fungi38PW_';
GRANT ALL
  ON fungi.*
  TO 'fungisuper'@'%.net';

CREATE USER 'fungisuper'@'%'
  IDENTIFIED BY 'fungi38PW_';
GRANT ALL
  ON fungi.*
  TO 'fungisuper'@'%';

  
CREATE USER 'fungisuper'@'localhost'
  IDENTIFIED BY 'fungi38PW_';
GRANT ALL
  ON fungi.*
  TO 'fungisuper'@'localhost';



CREATE USER 'fungiuser'@'localhost'
  IDENTIFIED BY 'fungipw';
GRANT ALL
  ON fungi.*
  TO 'fungiuser'@'localhost';
  
  
CREATE USER 'fungiuser'@'fungi.compute.dtu.dk'
  IDENTIFIED BY 'fungipw';
GRANT ALL
  ON fungi.*
  TO 'fungiuser'@'fungi.compute.dtu.dk';

(https://dev.mysql.com/doc/mysql-getting-started/en/)

 ALTER USER 'root'@'localhost' IDENTIFIED BY 'new_password';
 
## Fungi Design

# Data
image_id varchar(200) (key)
label (int)
set (train, test, test_final)

# label info
label (key)
species name varchar(200)

# Teams
Name varchar(200) (key)
active bool
password
credit_max

# requested image_labels
image_id (foreign key)
team (foreign key)
request_time

# submitted_labels
image_id (foreign key)
teams (foreign key)
label
submission_time

## Fungi functions

Initially, the requested image_labels is set so all teams have acces to a set of XXX image_labels

# return the image ids and labels that the team is available to the team
[image_ids, labels] = get_image_ids_and_labels(team, password)

# request the labels of the specified images
# returns None for images that are not in the database or not in the training set
# also remembers that the team has requested it (costs credits)
[labels] = request_labels(team, password, [image_ids])

# get image ids of data set (and if labels if any)
# training, training_with_labels, requested_labels, test, final
[image_ids, labels] = get_data_set(team, password, dataset)

# Get the number of available credits
credits = get_current_credits(team, password)

# submit the current classification result per image.
# will remember the submission time and also activates the team
# current classification scores are based on latest submission times
success = submit_label([images_ids], [labels], team, password)

# manager functions
[teams] = get_active_teams()

# get ground truth labels of the specified set (train, test, final_test)
[image_ids, labels] get_ground_truth_labels(set)

# get latest submitted labels per team for a given set
[images_ids, labels] get_latest_submitted_labels(set, team)


## Data splitting

183 different classes

32500 samples

40% in training without labels
10% in training with labels
25% in test
25% in final

for each class:
 find all ids of that class
 random shuffle
 40% first for train ('train')
 10% first for train with labels ('train_labels')
 25% next for test ('test')
 the rest in final ('final')
 

