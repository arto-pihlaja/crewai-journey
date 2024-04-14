from crewai import Agent, Crew, Process, Task
#from crewai_tools import SerperDevTool
from datetime import datetime
#from langchain.tools import DuckDuckGoSearchRun
#from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
#from langchain.utilities import GoogleSerperAPIWrapper

# Topic that will be used in the crew run
topic = 'Holiday daily plans'
destination = 'Nice, France'
start_date = '2024-07-01'
end_date = '2024-07-03'
number_of_days = datetime.strptime(end_date, "%Y-%m-%d") -\
datetime.strptime(start_date, "%Y-%m-%d")

# LangChain community DuckDuckGo search tool
search_tool = DuckDuckGoSearchRun()
#search_tool = SerperDevTool()
#search_tool = GoogleSerperAPIWrapper()
#search = GoogleSerperAPIWrapper()

# Creating first agent
goal = f"""Search for varied and entertaining family holiday activities."""
activity_guide = Agent(
    role='Activity ideator',
    goal=goal,
    verbose=True,
    backstory=f"""You are very good at finding activities. 
  You check facts meticulously before giving recommendations.""",
    tools=[search_tool],
    max_iter=20,
)

# Creating additional agents
restaurant_guide = Agent(
    role='Restaurant selector',
    goal=f'''Find restaurants for lunch and restaurants for dinner.''',
    verbose=True,
    backstory=
    """You excel at finding mid-market family friendly restaurants.""",
    tools=[search_tool],
    max_iter=15)

scheduler = Agent(
    role='Scheduler',
    goal=f"""Using a list of activities and and a list of restaurants, 
  create daily schedules 
  containing a morning activity, lunch, afternoon activity and dinner""",
    verbose=True,
    backstory=
    """You fit activity and restaurant recommendations into daily programs.""")

# Search
activity_search_task = Task(
    description=f"""
  Find {number_of_days * 2} leisure activities 
  within 20 kilometers from {destination}. 
  The activities must be available between {start_date} and {end_date} 
  and have positive reviews in tripadvisor.com
  and cost no more than 100 USD per person.
  Include the source URL of the information.""",
    expected_output=f"""YAML list of activities with a name, short description 
  and url for each.
  ---
  Example:
    - name: Castle Hill of Nice
      description: Castle Hill is a must-see attraction. It has ruins of a defensive wall, an artificial waterfall, and a elevator built into the rock.
      url: https://frenchriviera.travel/castle-hill-nice/
    - name: Old Town
      description: Nice Old Town is a characterful district in the south of the city. Probably the most famous tourist attraction in Old Nice is the busy market. 
      url: https://fi.hotels.com/go/france/old-town-nice
  """,
    max_inter=3,
    tools=[search_tool],
    agent=activity_guide)

# Research task for identifying AI trends
restaurant_search_task = Task(
    description=f"""Find {number_of_days * 2} restaurants for quick lunches 
  and {number_of_days * 2} restaurants for casual dinners. The restaurants 
  should be within 10 kilometers of {destination}. 
  The restaurants should have positive reviews. 
  """,
    expected_output=
    f"""YAML list of restaurants with a type, name and url for each.
  ---
  Example:
    - type: lunch
      name: La Pizza Cresci
      url: https://www.pizza-cresci-nice.com/fr/menus-carte.html
      description: Castle Hill, which towers over Nice’s historic core, is the most popular park in town.
      - type: dinner
        name: Le Clin d'Oeil
        url: https://www.tripadvisor.com/Restaurant_Review-g187234-d19275858-Reviews-Le_Clin_D_Oeil-Nice_French_Riviera_Cote_d_Azur_Provence_Alpes_Cote_d_Azur.html?m=19905
  """,
    max_inter=3,
    tools=[search_tool],
    agent=restaurant_guide)

# Scheduling task based on activity and restaurant findings
schedule_task = Task(
    context=[activity_search_task, restaurant_search_task],
    description=f"""You use the list of activities and list of restaurants 
  to create daily schedules around {destination} between 
  {start_date} and {end_date}. 
  Each schedule consists of morning activity, lunch, afternoon activity and dinner.""",
    expected_output=f"""One daily schedule per day between between 
  {start_date} and {end_date}. 
  Example of a daily schedule:
  ---
  Day 1:
  - morning activity: walk up to Castle Hill of Nice,  https://frenchriviera.travel/castle-hill-nice/ 
  - lunch: La Pizza Cresci, https://www.pizza-cresci-nice.com/fr/menus-carte.html
  - afternoon activity: visit old town of Nice,  https://fi.hotels.com/go/france/old-town-nice 
  - dinner: Le Clin d'Oeil, https://www.tripadvisor.com/Restaurant_Review-g187234-d19275858-Reviews-Le_Clin_D_Oeil-Nice_French_Riviera_Cote_d_Azur_Provence_Alpes_Cote_d_Azur.html?m=19905
  ---""",
    max_inter=3,
    agent=scheduler)

# Forming the planner crew
crew = Crew(
    agents=[activity_guide, restaurant_guide, scheduler],
    tasks=[activity_search_task, restaurant_search_task, schedule_task],
    process=Process.sequential  # Sequential task execution
)

result = crew.kickoff()
print(result)
