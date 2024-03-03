import streamlit as st
import plotly.express as px
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import datetime as dt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(
    page_title="F1 Data Analysis",
    page_icon="./images/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        # 'Get Help': '',
        # 'Report a bug': "",
        'About': "### Thanks for checking out the F1 Data Analysis App!"
    }
)


col1, mid, col2 = st.columns([1,1,20])
with col1:
    st.image('./images/favicon.png', width=100)
with col2:
    st.title('Formula 1 Data Analysis')

@st.cache_data(ttl=2*3600)
def get_session_data(year,gp,session):
    data = fastf1.get_session(year,gp,session)
    data.load()
    return data

@st.cache_data
def get_event_schedule_data(year):
    schedule = fastf1.get_event_schedule(year)
    schedule['EventDate'] = pd.to_datetime(schedule['EventDate'])
    today = dt.datetime.now()
    schedule['Days from Today'] = (schedule['EventDate'] - today).dt.days
    return schedule 

@st.cache_data
def get_event_data(year,gp):
    event = fastf1.get_event(year,gp)
    return event

# Sidebar

current_year = dt.datetime.now().year

events = get_event_schedule_data(current_year)
past_events = events[events['Days from Today'] < 0]

if len(past_events) == 0:
    past_events = get_event_schedule_data(current_year-1)
    current_year -= 1

year = st.sidebar.selectbox(
    "Year",
    np.arange(current_year,2020,-1)
)
events = get_event_schedule_data(year)
past_events = events[events['Days from Today'] < 0]
latest_event = len(past_events) - 1

gp = st.sidebar.selectbox(
    "Grand Prix",
    events.sort_values('EventDate').EventName.to_list(),
    index = latest_event
)
event = get_event_data(year,gp)
session = st.sidebar.selectbox(
    "Session",
    [event.get_session_name(n) for n in range(5,0,-1)]
)
try:
    data = get_session_data(year,gp,session)
    all_drivers = data.results.Abbreviation.unique()
    drivers = st.sidebar.multiselect(
        "Driver(s)",
        all_drivers,
        default=data.laps.groupby('Driver').LapTime.min().sort_values().reset_index()['Driver'].iloc[:2].to_list()
    )
    if len(drivers) == 0:
        drivers = all_drivers

    display_data_flag = 1

except:
    display_data_flag = 0
    st.warning("That event hasn't happened yet or doesn't exist! Please try again!",icon="⚠️")

footer="""<style>

.footer {
position: fixed;
bottom: 0;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by Srikrishnaa J</p>
</div>
"""
st.sidebar.markdown(footer,unsafe_allow_html=True)


#Functions

def convert_str_date_to_time(date):
    if date != 'NaT':
        if not '.' in date:
            date += '.000000'
        
        temp = dt.datetime.strptime(date,"0 days %H:%M:%S.%f")
        return dt.datetime.strftime(temp, "%H:%M:%S.%f")
    
    return 'No Time'

def convert_timedelta_to_time(date):
    if pd.isnull(date):
        return date
    out = str(date.seconds) + '.' + str(date.microseconds*1000)
    return float(out)

def basic_plots(drivers):
    df = pd.DataFrame()
    plots = [
        'Speed',
        # 'Distance',
        'Throttle',
        'DRS',
        'nGear'
        ]
    
    lap_time = {}
    for d in drivers:
        driver_df = data.laps.pick_driver(d).pick_fastest().get_car_data().add_distance()
        driver_df['Time'] = [round(n.total_seconds(),3) for n in driver_df['Time']]
        driver_df['Driver'] = d
        df = pd.concat([df,driver_df],axis=0)
    
        lap_time[d] = convert_timedelta_to_time(data.laps.pick_driver(d).pick_quicklaps().sort_values('LapTime').iloc[0]['LapTime'])

        # #Telemetry
        # tel_df = data.laps.pick_driver(d).pick_fastest().get_telemetry()
        # x = np.array(tel_df['X'].values)
        # y = np.array(tel_df['Y'].values)
        # points = np.array([x, y]).T.reshape(-1, 1, 2)
        # segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # shapes = [dict(
        #     type='line',
        #     x0=segments[i][0][0],
        #     y0=segments[i][0][1],
        #     x1=segments[i][1][0],
        #     y1=segments[i][1][1],
        #     line = dict(
        #     width=4,
        #     color='white'
        #     )
        # )
        # for i in range(segments.shape[0])]
        # # layout = go.Layout(
        # #     shapes=shapes,
        # #     title='Track'
        # # )
        # layout = {
        #     'xaxis': {
        #     # 'range': [0.2, 1],
        #     'showgrid': False, # thin lines in the background
        #     'zeroline': False, # thick line at x=0
        #     'visible': False  # numbers below
        #     },
        #     'yaxis': {
        #     # 'range': [0.2, 1],
        #     'showgrid': False, # thin lines in the background
        #     'zeroline': False, # thick line at x=0
        #     'visible': False  # numbers below
        #     }
        # }
        # data1 = go.Scatter(
        #     x=x,
        #     y=y,
        #     mode='lines',
        #     opacity=1,
        #     marker=dict(color='red',size=4)
        # )
        # fig = go.Figure([data1],layout)
        # fig.update_layout(autosize=False,width=500,height=500)
        # st.plotly_chart(fig,theme="streamlit")

    
    #Lap Time
    lap_time = pd.DataFrame(lap_time.values(),columns=['LapTime'],index=lap_time.keys()).sort_values('LapTime').reset_index()
    lap_time.rename({'index':'Driver'},axis=1,inplace=True)
    # st.dataframe(lap_time,use_container_width=True)
    for i,col in enumerate(st.columns(min(len(lap_time),5))):
        with col:
            minutes = int(lap_time['LapTime'].iloc[i]//60)
            seconds = round(lap_time['LapTime'].iloc[i]%60)
            milli = int(str(round(lap_time['LapTime'].iloc[i]%60,3)).split('.')[-1].strip())
            st.metric(label=lap_time['Driver'].iloc[i],value=f"{minutes:>02d}:{seconds:>02d}.{milli:<03d}")

    #Plots
    for i, p in enumerate(plots):
        fig = px.line(df,x='Time',y=f'{p}',color='Driver',title=p,color_discrete_sequence=[f'{fastf1.plotting.driver_color(d)}' for d in drivers])

        st.plotly_chart(fig,theme="streamlit",use_container_width=True)
    


def lap_times_plot(drivers):
    for d in drivers:
        df = data.laps.pick_driver(d).pick_quicklaps()
        
        df['LapTime'] = df['LapTime'].apply(convert_timedelta_to_time)
        stints = df[["Driver", "Stint", "Compound", "LapNumber"]].copy()
        stints = stints.groupby(["Driver", "Stint", "Compound"])
        stints = stints.count().reset_index()
        stints = stints.rename(columns={"LapNumber": "StintLength"})
        pit_stops = df[~df.PitInTime.isna()].LapNumber.to_list()
        fig = px.scatter(df,x='LapNumber',y='LapTime',color='Compound',title=f'{d} Lap Times at the {year} {gp} {session}',color_discrete_sequence=[fastf1.plotting.COMPOUND_COLORS[n] for n in df.Compound.unique()])
        for p in pit_stops:
            fig.add_vline(x=p,line_width=3,line_dash='dash',line_color=fastf1.plotting.driver_color(d))
        st.plotly_chart(fig,theme="streamlit",use_container_width=True)
        # fig = px.bar(
        #         stints,
        #         y='Driver',
        #         x='StintLength',
        #         color='Compound',
        #         color_discrete_sequence=[fastf1.plotting.COMPOUND_COLORS[n] for n in stints.Compound.unique()],
        #         orientation='h',
        #         opacity=1
        # )
        # st.plotly_chart(fig,theme="streamlit",use_container_width=True)

def rotate(xy, *, angle):
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    return np.matmul(xy, rot_mat)

def plot_speed_segments(drivers,fastest_lap=True):
    circuit_info = data.get_circuit_info()
    track_angle = circuit_info.rotation / 180 * np.pi
    lap = data.laps.pick_fastest()
    pos = lap.get_telemetry()
    dist_segments = circuit_info.marshal_lights.Distance.to_list()
    if dist_segments[0] > dist_segments[-1]:
        first = 0
        last = dist_segments[0]
        dist_segments[0] = first
        dist_segments.append(last)
    pos['dist_segments'] = pd.cut(pos.Distance,bins=dist_segments)
    driver_df = pd.Series()
    lap_time = {}
    for d in drivers:
        lap_time[d] = convert_timedelta_to_time(data.laps.pick_driver(d).pick_quicklaps().sort_values('LapTime').iloc[0]['LapTime'])
        if fastest_lap:
            temp_df = data.laps.pick_driver(d).pick_fastest().get_telemetry()
        else:
            temp_df = data.laps.pick_driver(d).pick_quicklaps().get_telemetry()
        temp_df['dist_segments'] = pd.cut(temp_df.Distance,bins=dist_segments)
        temp_df = temp_df.groupby('dist_segments')['Speed'].mean().reset_index()
        temp_df['Driver'] = d
        driver_df = pd.concat([driver_df,temp_df])

    max_speeds = driver_df.groupby('dist_segments')['Speed'].max().reset_index()
    driver_df = pd.merge(driver_df,max_speeds,on=['dist_segments','Speed'],how='inner')
    driver_df = driver_df.sort_values('dist_segments')
    pos = pd.merge(driver_df,pos,on=['dist_segments'],how='right')
    track = pos.loc[:, ('X', 'Y')].to_numpy()
    rotated_track = rotate(track, angle=track_angle)
    pos['X'] = rotated_track[:, 0]
    pos['Y'] = rotated_track[:, 1]
    pos['Driver_Colors'] = pos['Driver'].map(lambda x: fastf1.plotting.driver_color(x) if isinstance(x,str) else None)
    fig = go.Figure()
    for d in driver_df.Driver.unique():
        dom_segments = driver_df[driver_df['Driver']==d]['dist_segments'].unique()
        sub_pos = pos[pos['dist_segments'].isin(dom_segments)]
        for ds in sub_pos.dist_segments.unique():
            plot_pos = sub_pos[sub_pos['dist_segments']==ds]
            fig.add_trace(
                go.Scatter(x=plot_pos['X'],y=plot_pos['Y'],mode='lines',line=dict(color=fastf1.plotting.driver_color(d),width=10),hoverinfo='skip')
            )
            fig['data'][-1]['showlegend']=False
    title = f'Track Dominance {year} {gp} {session}'
    if fastest_lap:
        title += ' Fastest Lap Comparison'
    else:
        title += ' Throughout the Session'
    fig.update_layout(title=title,xaxis=dict(visible=False),
                           yaxis=dict(visible=False),
                           width=900,height=900,
                           plot_bgcolor='rgba(0,0,0,0)',paper_bgcolor='rgba(0,0,0,0)')
    
    offset_vector = [500, 0] 
    # Iterate over all corners.
    for _, corner in circuit_info.corners.iterrows():
        # Create a string from corner number and letter
        txt = f"{corner['Number']}{corner['Letter']}"

        # Convert the angle from degrees to radian.
        offset_angle = corner['Angle'] / 180 * np.pi

        # Rotate the offset vector so that it points sideways from the track.
        offset_x, offset_y = rotate(offset_vector, angle=offset_angle)

        # Add the offset to the position of the corner
        text_x = corner['X'] + offset_x
        text_y = corner['Y'] + offset_y

        # Rotate the text position equivalently to the rest of the track map
        text_x, text_y = rotate([text_x, text_y], angle=track_angle)

        # Rotate the center of the corner equivalently to the rest of the track map
        track_x, track_y = rotate([corner['X'], corner['Y']], angle=track_angle)

        # Draw a circle next to the track.
        # plt.scatter(text_x, text_y, color=circle_color, s=140)

        fig.add_trace(
                go.Scatter(x=[text_x],y=[text_y],mode='text',text=txt,textposition='middle center',hoverinfo='skip',textfont=dict(size=20))
            )
        fig['data'][-1]['showlegend']=False
        # Draw a line from the track to this circle.
        # plt.plot([track_x, text_x], [track_y, text_y], color=line_color)

        # Finally, print the corner number inside the circle.
        # plt.text(text_x, text_y, txt,
        #         va='center_baseline', ha='center', size='small', color=text_color)
    
    for d in drivers:
        fig.add_trace(go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    name=d,
                    line=dict(color=fastf1.plotting.driver_color(d)),
                ))
        fig.update_traces(dict(showlegend=True),selector=({'name':d}))
    lap_time = pd.DataFrame(lap_time.values(),columns=['LapTime'],index=lap_time.keys()).sort_values('LapTime').reset_index()
    lap_time.rename({'index':'Driver'},axis=1,inplace=True)

    for i,col in enumerate(st.columns(min(len(drivers),5))):
        with col:
            minutes = int(lap_time['LapTime'].iloc[i]//60)
            seconds = round(lap_time['LapTime'].iloc[i]%60)
            milli = int(str(round(lap_time['LapTime'].iloc[i]%60,3)).split('.')[-1].strip())
            st.metric(label=lap_time['Driver'].iloc[i],value=f"{minutes:>02d}:{seconds:>02d}.{milli:<03d}")
            st.markdown(f'<h4 style="color:{fastf1.plotting.driver_color(drivers[i])}">{drivers[i]}</h4>',unsafe_allow_html=True)
    st.plotly_chart(fig,theme="streamlit",use_container_width=True)


#Tabs
if display_data_flag:
    tab1, tab2, tab3, tab4 = st.tabs(["Results","Fastest Comparison","Track Dominance", "Lap By Lap"])


    #       Tab 1
    with tab1:
        # data = get_session_data()
        results = data.results
        if 'Practice' in session:
            results = pd.merge(right=data.laps.groupby('Driver').LapTime.min().sort_values().reset_index(),left=data.results,right_on='Driver',left_on='Abbreviation',how='outer')
            results['Fastest Time'] = results['LapTime']
            results['LapTime'] = results['LapTime'].apply(convert_timedelta_to_time)
            results = results.sort_values('LapTime')
            results['Fastest Time'] = results['Fastest Time'].astype(str)
            results['Fastest Time'] = results['Fastest Time'].apply(convert_str_date_to_time)
        
        st.header(f'{year} {gp} {session} Results')
        
        def path_to_image_html(path):
            return '<img src="' + path + '">'
        
        results['Headshot'] = results['HeadshotUrl'].map(path_to_image_html)
        results['TeamColor'] = results['TeamColor'].apply(lambda x: '#'+x)

        results['Time'] = results['Time'].astype(str)
        results['Time'] = results['Time'].apply(convert_str_date_to_time)

        results['Q1'] = results['Q1'].astype(str)
        results['Q1'] = results['Q1'].apply(convert_str_date_to_time)

        results['Q2'] = results['Q2'].astype(str)
        results['Q2'] = results['Q2'].apply(convert_str_date_to_time)

        results['Q3'] = results['Q3'].astype(str)
        results['Q3'] = results['Q3'].apply(convert_str_date_to_time)

        cols = ['Headshot', 'Abbreviation','FullName',  'TeamName',]
        if session == 'Race' or session == 'Sprint':
            add_cols = ['ClassifiedPosition','GridPosition','Time','Status','Points']
            for a in add_cols:
                cols.append(a)
        elif session == 'Qualifying':
            add_cols = ['Q1','Q2','Q3']
            for a in add_cols:
                cols.append(a)
        
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
        st.markdown(html,unsafe_allow_html=True)


    #       Tab 2
    with tab2:
        st.header(f'{year} {gp} {session} Fastest Lap Comparison')
        basic_plots(drivers)

    #       Tab 3
    with tab3:
        subtab1, subtab2 = st.tabs(["Fastest Lap","Full session"])
        with subtab1:
            st.header(f'{year} {gp} {session} Track Dominance Fastest Lap')
            plot_speed_segments(drivers,fastest_lap=True)
        with subtab2:
            st.header(f'{year} {gp} {session} Track Dominance Full Session')
            plot_speed_segments(drivers,fastest_lap=False)

    #       Tab 4
    with tab4:
        st.header(f'{year} {gp} {session} Lap by Lap Comparison')
        lap_times_plot(drivers)

    