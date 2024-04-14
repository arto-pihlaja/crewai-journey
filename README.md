# Simple CrewAI test

## Introduction
Spin up a Crew of agents to create daily plans of activities and meals on a leisure trip.

This project is based on CrewAI's [Getting Started Guide](https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/). Runs as a Repl. 

The sample code from crewai.com doesn't work out of the box, but must be updated to use newer versions of langchain, crewai and duckduckgo-search. Currently working set of versions in lock file.

## Running the Script
You need to add your OpenAI API key to the `OPENAI_API_KEY` variable in the Secrets tool.

CrewAI use gpt-4 by default unless you change the `OPENAI_MODEL_NAME` variable in the Secrets tool. 

## Improvement initiatives
<li>save results to file or add ui</li>
<li>add an agent to verify URLS and remove dead ones

## License
This project is released under the MIT License.
