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
activity_goal = f'Find family activities for a holiday in {destination}.'
activity_guide = Agent(
  role='Activity ideator',
  goal=activity_goal,
  verbose=True,
  backstory=f"""You are very good at finding varied holiday activities. You search
  for {number_of_days * 2} activities within 20 kilometers from {destination} and available between
  between {start_date} and {end_date}' and cost no more than 50 USD per person.""",
  tools=[search_tool]
)

# Creating additional agents
restaurant_guide = Agent(
  role='Restaurant selector',
  goal=f'''Find {number_of_days * 2} restaurants for lunch and dinner around {destination} 
  between {start_date} and {end_date}''',
  verbose=True,
  backstory="""You are a helpful agent that finds family friendly restaurants 
  within a range of 20 kilometers from the destination."""
)

scheduler = Agent(
  role='Scheduler',
  goal=f'Using a list of activities and and a list of restaurants, create daily schedules.',
  verbose=True,
  backstory="""You plan daily schedules containing a morning activity, lunch, 
  afternoon activity and dinner.  The activities and restaurants on one day should 
  be less than 10 kilometers apart."""
)


# Search
activity_search_task = Task(
  description=f"""Find activities within 20 kilometers of {destination}. The activities
  should be available between {start_date} and {end_date} 
  and have positive reviews in tripadvisor.com. 
  """,
  expected_output=f"""YAML list of activities with a name, short description 
  and url for each.
  ---
  Example:
    - name: Castle Hill of Nice
      description: Castle Hill, which towers over Nice’s historic core, is the most popular park in town.
      url: https://www.tripadvisor.com/Attraction_Review-g187234-d247503-Reviews-Castle_Hill_of_Nice-Nice_French_Riviera_Cote_d_Azur_Provence_Alpes_Cote_d_Azur.html
    - name: Old Town
      description: This historic part of Nice feels like a medieval village with narrow streets curving between old buildings
      url: https://frenchriviera.travel/old-town-nice/
  """,
  max_inter=3,
  tools=[search_tool],
  agent=activity_guide
)

# Research task for identifying AI trends
restaurant_search_task = Task(
  description=f"""Find restaurants within 10 kilometers of {destination}. 
  Half of the restaurants should be good for quick lunches, 
  half for dinners. The restaurants should have received positive reviews. 
  """,
  expected_output=f"""YAML list of restaurants with a type, name and url for each.
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
  agent=restaurant_guide
)

# Scheduling task based on activity and restaurant findings
schedule_task = Task(
  description=f"""Create daily schedules around {destination} between 
  {start_date} and {end_date}.""",
  expected_output=f"""One daily schedule per day between between 
  {start_date} and {end_date}. 
  Example of a daily schedule:
  ---
  Day 1:
  - morning activity: walk up to Castle Hill of Nice, https://www.tripadvisor.com/Attraction_Review-g187234-d247503-Reviews-Castle_Hill_of_Nice-Nice_French_Riviera_Cote_d_Azur_Provence_Alpes_Cote_d_Azur.html
  - lunch: La Pizza Cresci, https://www.pizza-cresci-nice.com/fr/menus-carte.html
  - afternoon activity: visit old town of Nice, https://www.tripadvisor.com/Attraction_Review-g187234-d254286-Reviews-Old_Town-Nice_French_Riviera_Cote_d_Azur_Provence_Alpes_Cote_d_Azur.html
  - dinner: Le Clin d'Oeil, https://www.tripadvisor.com/Restaurant_Review-g187234-d19275858-Reviews-Le_Clin_D_Oeil-Nice_French_Riviera_Cote_d_Azur_Provence_Alpes_Cote_d_Azur.html?m=19905
  ---""",
  max_inter=3,
  tools=[search_tool],
  agent=scheduler
)

# Forming the tech-focused crew
crew = Crew(
  agents=[activity_guide, restaurant_guide, scheduler],
  tasks=[activity_search_task, restaurant_search_task, schedule_task],
  process=Process.sequential  # Sequential task execution
)

result = crew.kickoff()
print(result)
