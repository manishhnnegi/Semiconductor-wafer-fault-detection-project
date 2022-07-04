import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
import pandas as pd
from application_logging.logger import App_Logger

# from application_logging.logger import App_Logger


class dBOperation:
    """
      This class shall be used for handling all the SQL operations.

      Written By: iNeuron Intelligence
      Version: 1.0
      Revisions: None

      """

    def __init__(self):
        self.path = 'Training_Database/'
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Logger()

    def dataBaseConnection(self, DatabaseName):

        """
                Method Name: dataBaseConnection
                Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
                Output: Connection to the DB
                On Failure: Raise ConnectionError

                 Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """
        try:
            conn = sqlite3.connect(self.path + DatabaseName + '.db')
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Opened %s database successfully" % DatabaseName)
            file.close()

        except ConnectionError:
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Error while connecting to database: %s" % ConnectionError)
            file.close()

            raise ConnectionError
        return conn

    def insertIntoTableGoodData(self, Database):

        """
                               Method Name: insertIntoTableGoodData
                               Description: This method inserts the Good data files from the Good_Raw folder into the
                                            above created table.
                               Output: None
                               On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

        """

        conn = self.dataBaseConnection(Database)
        goodFilePath = self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]
        log_file = open("Training_Logs/DbInsertLog.txt", 'a+')
        try:
            for file in onlyfiles:
                with open(goodFilePath + '/' + file, "r") as f:
                    df = pd.read_csv(f)
                    df.to_sql('Good_Raw_Data', conn, if_exists='append', index=False)
                    self.logger.log(log_file, " %s: File loaded successfully!!" % file)
                    conn.commit()

        except Exception as e:
            conn.rollback()
            self.logger.log(log_file, "Error while creating table: %s " % e)
            shutil.move(goodFilePath + '/' + file, badFilePath)
            self.logger.log(log_file, "File Moved Successfully %s" % file)
            log_file.close()
            conn.close()

        conn.close()
        log_file.close()





    def selectingDatafromtableintocsv(self, Database):

        """
                               Method Name: selectingDatafromtableintocsv
                               Description: This method exports the data in GoodData table as a CSV file. in a given location.
                                            above created .
                               Output: None
                               On Failure: Raise Exception

                                Written By: iNeuron Intelligence
                               Version: 1.0
                               Revisions: None

        """
        fileFromDb = 'Training_FileFromDB/'
        fileName = 'InputFile.csv'
        log_file = open("Training_Logs/ExportToCsv.txt", 'a+')
        try:

            # Make the CSV ouput directory
            if not os.path.isdir(fileFromDb):
                os.makedirs(fileFromDb)

            conn = self.dataBaseConnection(Database)
            c = conn.cursor()
            df = pd.read_sql_query("select * from Good_Raw_Data ", conn)

            df.to_csv(fileFromDb + fileName, index=None, header=True)
            #c.execute('drop table Good_Raw_Data')
            self.logger.log(log_file, "File exported successfully!!!")
            log_file.close()

            conn.close()


        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" % e)
            log_file.close()






