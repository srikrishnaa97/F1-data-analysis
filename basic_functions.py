import datetime as dt 
import pandas as pd
import numpy as np

def convert_str_date_to_time(date):
    if date != 'NaT':
        if not '.' in date:
            date += '.000000'

        temp = dt.datetime.strptime(date, "0 days %H:%M:%S.%f")
        return dt.datetime.strftime(temp, "%H:%M:%S.%f")

    return 'No Time'


def convert_timedelta_to_time(date):
    if pd.isnull(date):
        return date
    out = str(date.seconds) + '.' + str(date.microseconds * 1000)
    return float(out)

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)