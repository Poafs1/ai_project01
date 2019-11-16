# ai_project01
## Instruction
To use the template, you will have to install 4 additional packages: <br/>
* [Numpy](https://numpy.org)
* [OpenAI's Gym](https://github.com/openai/gym)
* [tqdm](https://tqdm.github.io)
* [boardgame2](https://pypi.org/project/boardgame2/)

In this project, you will work on an AI for Othello (or Reversi) game. You should form a team of 2-3 members and form a division of 4 teams. Each team will implement: <br/>

* An evaluation function
* Depth-limited Minimax with Alpha-Beta pruning

Within a division, you can collaborate on the coding. However, you should keep in mind that the winning team in a division gets 1 point. Consulting an internet is okay, but you need to cite your source. <br/>

You will organize an Othello league within your division and select one best team to compete within your section. In the end, we will run a tournament for each section and for all sections. <br/>

## Competition Rules
* Each turn, your AI has only 10 seconds to provide a move, otherwise, a random move will be made.
* Your AI must not search beyond 10 steps.
* Your evaluation function should not perform searching (e.g. Monte Carlo simulation).

## Submission
Zip your submission and name it as __sec<x>-<d>-othello-project.zip,__ where __x__ is your section number and __d__ is your division name. The zip file should contain the following: <br/>
  
* The code of the division winner.
* The result of the competition in the following format:

| Black/White | Team 1 | Team 2 | Team 3 | Team 4 |
| --- | --- | --- | --- | --- |
| __Team 1__ | - | 45 - 10 | ... | ... |
| __Team 2__ | 30 - 20 | - | ... | ... |
| __Team 3__ | ... | ... | - | ... |
| __Team 4__ | ... | ... | ... | - |

### Due date
Before the end of Nov 27, 2019, Thailand time. <br/>

### Scoring
This project has a total score of 10% of the class. Your project will be graded based on the following criteria:

* [7 points] The correctness of your implementation: <br/>
[2 points] Evaluation function. <br/>
[3 points] Alpha-Beta search. <br/>
[1 point] Depth-limited condition. <br/>
[1 point] Action ordering (to make pruning more effective). <br/>
* [3 points] Result of the competition: <br/>
[1 point] for everyone in a winning team of a division. <br/>
[1 point] for everyone in a division that wins the section-level tournament. <br/>
[1 point] for everyone in a section that wins the 3rd-year tournament. <br/>






