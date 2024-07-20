import datetime as dt 
import pandas as pd
import numpy as np

def convert_str_date_to_time(date):
    if date != 'NaT' and date != 'No Time':
        if not '.' in date:
            date += '.000000'

        try: 
            temp = dt.datetime.strptime(date, "0 days %H:%M:%S.%f")
        except:
            temp = dt.datetime.strptime(date, "%H:%M:%S.%f")
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

def path_to_image_html(path):
    return '<img src="' + path + '">'

def get_results_df_as_html(data):
    results = data.results
    session = data.session_info['Name']
    if 'Practice' in session:
        results = pd.merge(right=data.laps.groupby('Driver').LapTime.min().sort_values().reset_index(),
                            left=data.results, right_on='Driver', left_on='Abbreviation', how='outer')
        results['Fastest Time'] = results['LapTime']
        results['LapTime'] = results['LapTime'].apply(convert_timedelta_to_time)
        results = results.sort_values('LapTime')
        results['Fastest Time'] = results['Fastest Time'].astype(str)
        results['Fastest Time'] = results['Fastest Time'].apply(convert_str_date_to_time)
        results.sort_values('Fastest Time',inplace=True)


    results['Headshot'] = results['HeadshotUrl'].map(path_to_image_html)
    results['TeamColor'] = results['TeamColor'].apply(lambda x: '#' + x)

    results['Time'] = results['Time'].astype(str).apply(convert_str_date_to_time)

    results['Q1'] = results['Q1'].astype(str).apply(convert_str_date_to_time)
    results['Q2'] = results['Q2'].astype(str).apply(convert_str_date_to_time)
    results['Q3'] = results['Q3'].astype(str).apply(convert_str_date_to_time)

    cols = ['Headshot', 'Abbreviation', 'FullName', 'TeamName', ]
    if session == 'Race' or session == 'Sprint':
        cols = cols + ['ClassifiedPosition', 'GridPosition', 'Time', 'Status', 'Points']
        results['GridPosition'] = results['GridPosition'].astype(str).map(lambda x: x.split('.')[0])
    elif session == 'Qualifying' or session == 'Sprint Qualifying' or session == 'Sprint Shootout':
        cols = cols + ['Q1', 'Q2', 'Q3']

    elif 'Practice' in session:
        cols.append('Fastest Time')

    else:
        pass

    html = '<table>'
    html += '<tr>'
    for c in cols:
        html += f'<th>{c}</th>'
    html += '</tr>'
    for i in range(len(results)):
        html += '<tr>'
        for c in cols:
            if c == 'TeamName':
                html += f'<td>{results[c].iloc[i]} <div style="background-color:{results["TeamColor"].iloc[i]};width:4vw;height:1vw;align:center;"></div></td>'

            elif (c == 'Time' and i == 0 and results[c].iloc[i] != 'No Time'):
                html += f'<td>{results[c].iloc[i][:-3]}</td>'

            elif (c == 'Time' and i > 0 and results[c].iloc[i] != 'No Time'):
                html += f'<td>+{results[c].iloc[i][3:-3]}</td>'

            elif c.startswith('Q') and results[c].iloc[i] != 'No Time':
                html += f'<td>{results[c].iloc[i][3:-3]}</td>'

            elif c == 'Fastest Time' and results[c].iloc[i] != 'No Time':
                html += f'<td>{results[c].iloc[i][3:-3]}</td>'

            else:
                html += f'<td>{results[c].iloc[i]}</td>'
        html += '</tr>'

    html += '</table>'
    return html