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

year = st.sidebar.radio(
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
            st.metric(label=lap_time['Driver'].iloc[i],value=f"{int(lap_time['LapTime'].iloc[i]//60)}:{round(lap_time['LapTime'].iloc[i]%60,3)}")

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




#Tabs
if display_data_flag:
    tab1, tab2, tab3 = st.tabs(["Results","Fastest Comparison","Lap By Lap"])


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
        st.header(f'{year} {gp} {session} Lap by Lap Comparison')
        lap_times_plot(drivers)

    