# IR-Index-Construction
Index Creation for Information Retrieval

Before you start, make sure to read the Search Engine Specification

Milestone 1: Index construction
Data sources:

(1) Information analyst: small collection of web pages

(2) Algorithms and data structures developer: larger collection of web pages

Building the inverted index:
Now that you have been provided the HTML files to index, you may build your inverted index off of them. The inverted index is simply a map with the token as a key and a list of its corresponding postings. A posting is the representation of the token’s occurrence in a document. The posting typically (not limited to) contains the following info (you are encouraged to think of other attributes that you could add to the index):

The document name/id the token was found in.
Its tf-idf score for that document (for MS1, add only the term frequency)
 Some tips:

When designing your inverted index, you will think about the structure of your posting first.
You would normally begin by implementing the code to calculate/fetch the elements which will constitute your posting.
Use scripts/classes that will perform a function or a set of closely related functions. This helps in keeping track of your progress, debugging, and also dividing work amongst teammates if you’re in a group.
We strongly recommend you use GitHub as a mechanism to work with your team members on this project, but please make the project private.
Deliverables: Submit your code and a report (in PDF format) with a table containing some analytics about your index. The minimum analytics are:  

The number of indexed documents;
The number of unique tokens;
The total size (in KB) of your index on disk.
Note for the developer option: at this time, you do not need to have the optimized index, but you may save time if you do. 
No late submissions will be accepted for this milestone.

Evaluation criteria:

Did your report show up on time?
Are the reported numbers plausible?
Important note: You can only change teams between the milestones of Assignment 3 if there are strong reasons for doing so. Please, write an email to the teaching team explaining why you need to change teams and we will assess your situation.
