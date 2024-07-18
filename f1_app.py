import streamlit as st
import plotly.express as px
import fastf1
import fastf1.plotting
import streamlit.components.v1 as components
import datetime as dt
import pandas as pd
import numpy as np
from plot_functions import basic_plots, lap_times_plot, plot_speed_segments, track_animation
from basic_functions import convert_str_date_to_time, convert_timedelta_to_time

# fastf1.Cache.enable_cache('./cache')
# fastf1.Cache.clear_cache()
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

col1, mid, col2 = st.columns([1, 1, 20])
with col1:
    st.image('./images/favicon.png', width=100)
with col2:
    st.title('Formula 1 Data Analysis')


@st.cache_data(ttl=2 * 3600,max_entries=3,show_spinner='Fetching session data...')
def get_session_data(year, gp, session):
    # _, cache_size = fastf1.Cache.get_cache_info()
    # if cache_size >= 3e8:
    #     fastf1.Cache.clear_cache()
    data = fastf1.get_session(year, gp, session)
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
def get_event_data(year, gp):
    event = fastf1.get_event(year, gp)
    return event


# Sidebar

current_year = dt.datetime.now().year

events = get_event_schedule_data(current_year)
past_events = events[events['Days from Today'] < 0]

if len(past_events) == 0:
    past_events = get_event_schedule_data(current_year - 1)
    current_year -= 1

year = st.sidebar.selectbox(
    "Year",
    np.arange(current_year, 2020, -1)
)
events = get_event_schedule_data(year)
past_events = events[events['Days from Today'] < 0]
latest_event = len(past_events) - 1

gp = st.sidebar.selectbox(
    "Grand Prix",
    events.sort_values('EventDate').EventName.to_list(),
    index=latest_event
)
event = get_event_data(year, gp)
session = st.sidebar.selectbox(
    "Session",
    [event.get_session_name(n) for n in range(5, 0, -1)]
)
try:
    data = get_session_data(year, gp, session)
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
    st.warning("That event hasn't happened yet or doesn't exist! Please try again!", icon="⚠️")

footer = """<style>

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
st.sidebar.markdown(footer, unsafe_allow_html=True)

event_data = {'year':year,'gp':gp,'session':session}


# Tabs
if display_data_flag:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Results", "Fastest Comparison", "Track Dominance", "Lap By Lap", "Track Animation"])

    #       Tab 1
    with tab1:
        # data = get_session_data()
        results = data.results
        if 'Practice' in session:
            results = pd.merge(right=data.laps.groupby('Driver').LapTime.min().sort_values().reset_index(),
                               left=data.results, right_on='Driver', left_on='Abbreviation', how='outer')
            results['Fastest Time'] = results['LapTime']
            results['LapTime'] = results['LapTime'].apply(convert_timedelta_to_time)
            results = results.sort_values('LapTime')
            results['Fastest Time'] = results['Fastest Time'].astype(str)
            results['Fastest Time'] = results['Fastest Time'].apply(convert_str_date_to_time)

        st.header(f'{year} {gp} {session} Results')


        def path_to_image_html(path):
            return '<img src="' + path + '">'


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
        st.markdown(html, unsafe_allow_html=True)

    #       Tab 2
    with tab2:
        st.header(f'{year} {gp} {session} Fastest Lap Comparison')
        figs, kpi_dict = basic_plots(data,event_data,drivers)
        for i, col in enumerate(st.columns(len(kpi_dict))):
            with col:
                driver = list(kpi_dict.keys())[i]
                value = kpi_dict[driver]
                st.metric(label=driver, value=value)
                st.markdown(f'<h4 style="color:{fastf1.plotting.driver_color(driver)}">{driver}</h4>',
                            unsafe_allow_html=True)

        for fig in figs:
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

    #       Tab 3
    with tab3:
        subtab1, subtab2 = st.tabs(["Fastest Lap", "Full session"])
        with subtab1:
            st.header(f'{year} {gp} {session} Track Dominance Fastest Lap')
            fig1, kpi_dict = plot_speed_segments(data, event_data, drivers, fastest_lap=True)
            for i, col in enumerate(st.columns(len(kpi_dict))):
                with col:
                    driver = list(kpi_dict.keys())[i]
                    value = kpi_dict[driver]
                    st.metric(label=driver, value=value)
                    st.markdown(f'<h4 style="color:{fastf1.plotting.driver_color(driver)}">{driver}</h4>',
                            unsafe_allow_html=True)
            st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
        with subtab2:
            st.header(f'{year} {gp} {session} Track Dominance Full Session')
            fig2, kpi_dict = plot_speed_segments(data, event_data, drivers, fastest_lap=False)
            st.plotly_chart(fig2,theme="streamlit",use_container_width=True)

    #       Tab 4
    with tab4:
        st.header(f'{year} {gp} {session} Lap by Lap Comparison')
        figs = lap_times_plot(data,event_data,drivers)
        for fig in figs:
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    
    #       Tab 5
    with tab5:
        st.header(f'{year} {gp} {session} Track Animation')
        fig = track_animation(data,drivers)
        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
