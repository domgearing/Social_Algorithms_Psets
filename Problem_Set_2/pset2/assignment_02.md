Assignment 2: Analyzing Bluesky
Due Date: Monday 2/16
Overview
In this assignment, you’ll analyze several aspects of Bluesky. You’ll construct reverse-chronological
feeds. You’ll make simple follow recommendations. And you’ll examine h om o ph i l y — the tendency
for people to associate with similar others — in Bluesky replies. You’ll collect data using Bluesky’s
public API, analyze what senators see in their feeds, and measure whether reply patterns exhibit
gender homophily.
The assignment has two parts:
Part I: Feed Analysis — Examine what U.S. senators see when they open Bluesky, and make
follow recommendations for fellow senators.
Part II: Reply Homophily — Measure gender patterns in who replies to senators
Background
Bluesky is a social media platform built on the open AT Protocol. Unlike Twitter/X, Bluesky’s API is
publicly accessible without authentication for read-only operations, making it valuable for studying
social network dynamics.
We focus on U.S. Senators with verified Bluesky accounts. As of January 2026, 42 senators have
verified accounts: 41 Democrats and 1 Independent (Bernie Sanders). No Republican senators
have verified Bluesky accounts. This partisan skew is itself a phenomenon worth reflecting on (see
Part III).
Provided Materials
Download all materials from the course website:
https://socialalgorithms.github.io/assignment2
_
bluesky 1/10
2/11/26, 12:31 PM senators
Assignment 2: Analyzing Bluesky | Social Algorithms
_
bluesky.csv — 42 verified senator accounts (columns: name, handle, party, state,
gender)
bluesky_
helpers.py — Helper functions and code scaffolding
female
_
names.tsv.gz — SSA baby name data for female names
male
_
names.tsv.gz — SSA baby name data for male names
The name files contain historical Social Security Administration baby name registration data in tab-
separated format ( name\tyear\tcount). These are useful for inferring gender from first names
based on historical naming patterns.
Part I: Feed Analysis
When a senator opens Bluesky, they see a reverse-chronological feed of posts from accounts they
follow. This feed represents their information environment on the platform.
I.1 Data Collection
Write code to collect the feed for each senator:
1. For each senator, retrieve the list of accounts they follow using getFollows
2. For each account in that follow list, fetch that account’s posts from the last 24 hours using
getAuthorFeed
3. Combine these posts into a single reverse-chronological feed for each senator
4. Save the data as JSON (one file per senator, or one combined file)
Note: Look ahead at Section I.3. You’ll need to track both the follow relationships and the posts (for
computing Jaccard similarities). Consider saving follow data and feed data separately.
API endpoints (see bluesky_
helpers.py for wrapper functions):
app.bsky.graph.getFollows — list of followed accounts (paginated, up to 100 per request)
app.bsky.feed.getAuthorFeed — recent posts from an account
For API documentation see: Bluesky API documentation
Rate limits: Add time.sleep(0.1) between requests. The public API allows ~3,000 requests per 5
minutes.
I.2 “Senators You May Know” Recommendations
https://socialalgorithms.github.io/assignment2
_
bluesky 2/10
2/11/26, 12:31 PM Assignment 2: Analyzing Bluesky | Social Algorithms
Build a simple recommendation system to suggest which senators each senator should follow,
based on the follow graph among senators. Use a triangle-counting approach: for each senator X
who doesn’t follow senator Y, count how many senators that X d o e s follow also follow Y. This is the
“mutual follows” count — the more senators in X’s network who follow Y, the stronger the
recommendation.
recommendation
_
score(A, B) = |{C : A follows C and C follows B}|
This is analogous to main feature in “people you may know” recommendations on various social
networks.
Task
For each senator:
1. Identify which other senators they do not currently follow
2. For each non-followed senator, compute the recommendation score (number of followed
senators who follow them)
3. Report the top 3 recommendations with their scores
If a senator already follows all other senators, note this in your output. If a senator follows no other
senators, note this in your output.
Output
Generate a table in your report (programmatically, not by hand) with the following format:
Senator Recommendation 1 Recommendation 2 Recommendation 3
Bernie Sanders Chuck Schumer (8) Amy Klobuchar (7) …
Elizabeth Warren f ol l ows al l s en a t or s
… … … …
The number in parentheses is the recommendation score (mutual follows count).
Additionally, identify:
Which senators (if any) follow all or follow zero other senators in the dataset
Which senators (if any) are followed by all or followed by zero other senators in the dataset
https://socialalgorithms.github.io/assignment2
_
bluesky 3/10
2/11/26, 12:31 PM Assignment 2: Analyzing Bluesky | Social Algorithms
(Note: patterns in recommendations, network centrality, and disconnected senators are the subject
of questions in Section I.4 below.)
I.3 Echo Chamber Analysis
Examine the degree to which senators’ feeds overlap using two complementary measures:
Follow Jaccard Similarity
For each pair of senators, compute the Jaccard similarity of the accounts they follow (all accounts,
not just senators):
Jaccard(A, B) = |follows
A ∩ follows
_
_
B| / |follows
A ∪ follows
_
_
B|
This measures overlap in terms of wh o the senators follow, treating all followed accounts equally
regardless of how active they are.
Post Jaccard Similarity
The follow Jaccard treats a dormant account the same as one posting 50 times per day, and doesn’t
necessarily reflect similarities in their user experience. To capture similarity in what senators
a c t ual l y s e e, define and compute a suitable Jaccard coefficient over post in their feed in the last 24
hours.
Each of these measures (follow Jaccard, post Jaccard) can be interpreted as similarity measures
between the senators. Computing these measures for all the senators in the dataset forms two
similarity matrices.
Visualization and analysis
1. Create two heatmaps: one for follow Jaccard similarity, one for post Jaccard similarity. You
may present these as two separate figures or side-by-side in a single figure for easy
comparison.
2. Sort the rows and columns so that similar senators are adjacent. Use hierarchical clustering
to determine the ordering (see scipy.cluster.hierarchy.linkage and dendrogram), or (if
you want to explore alternative visualization techniques) explore some other sorting
algorithm. An unsorted heatmap with arbitrary row/column order is much harder to interpret
— clusters appear scattered rather than as visible blocks along the diagonal. Use the same
ordering for both heatmaps to enable direct comparison.
https://socialalgorithms.github.io/assignment2
_
bluesky 4/10
2/11/26, 12:31 PM Assignment 2: Analyzing Bluesky | Social Algorithms
3. Compare and contrast the two matrices: What do they show? Interpret any clustering
patterns you see. Where do the two matrices differ? Senators with similar follows but different
posts might follow the same accounts but experience very different information environments
based on posting frequency.
Suggested questions to consider (you don’t need to answer all or any of these, but they may
guide your interpretation): Do senators from the same state follow similar accounts? Do female
senators follow systematically different accounts than male senators? Are there identifiable
“information bubbles”? How does weighting by post volume change the similarity structure?
I.4 Reflection Questions
Include brief responses to these questions in your report:
1. What does the overlap structure of senators’ feeds (from the Jaccard similarity analysis)
suggest about shared versus divergent information within the Democratic caucus? Are there
identifiable clusters or outliers?
2. A senator following 50 high-volume accounts might see 500 posts/day; one following 500
accounts might see 5,000. What are the implications of this variation for how senators use the
platform or stay informed about constituent concerns?
3. Based on your “Senators You May Know” analysis, which senators appear most frequently as
recommendations? What does this suggest about their position in the senate follow network?
Are there senators who are notably disconnected from their colleagues on Bluesky?
Part II: Reply Homophily Analysis
In this section we will look at the posts by senators themselves, and who replies to these posts. We
are going to specifically investigate homphily in the replies, meaning that women
disproportionately reply to female senators, and men disproportionately reply to male senators.
The key question here is how to think about “disproportionately”.
II.1 Data Collection
Write code to collect replies to senator posts. To ensure sufficient data for gender comparisons,
collect reply data for at least 5 female senators and at least 5 male senators (the dataset
contains 14 female and 28 male senators), uniformly at random. If you can, take the time to run
your code long enough to collect data for all the senators.
https://socialalgorithms.github.io/assignment2
_
bluesky 5/10
2/11/26, 12:31 PM For each senator in your sample:
Assignment 2: Analyzing Bluesky | Social Algorithms
1. Fetch their posts from the last 7 days using getAuthorFeed
2. For each post with replies, fetch the reply thread using getPostThread
3. Extract replier information (handle, display name, timestamp) and post metadata (reply count,
for understanding sampling)
4. Save the data (in JSON) for each senator
Important: Save the post’s total replyCount field along with the replies you captured. This lets you
determine what fraction of replies you collected for each post, which is essential for the timing
analysis.
Important limitation: The API returns at most ~200 replies per post, focused on the e arl i e s t
replies. For posts with 1,000+ replies, you will see only a fraction of the conversation. You must
consider this limitation in your analysis.
II.2 Gender Inference
Make an inference about the gender of repliers from their display names. Using the SSA baby name
data. For instance:
1. Extract a likely first name from each display name
2. Look up the name’s historical gender distribution
3. Classify as female if >60% of registrations are female, male if >60% are male, otherwise
unknown
Report what fraction of repliers you can classify and discuss the limitations of this approach. Feel
free to try to improve on this method (query an LLM?), but it is not a required part of the
assignment.
II.3 Homophily Measurement
Measure gender patterns in replies:
1. Baseline: Across all senators, what fraction of classifiable repliers are female vs. male? Call
these p_
female and p_
male = 1 - p_
female.
2. By senator gender:
For female senators, what fraction of their repliers are female? Call this obs
F.
_
For male senators, what fraction of their repliers are male? Call this obs
M.
_
3. Homophily coefficient: Compute a measure of same-gender homophily for each senator
gender:
https://socialalgorithms.github.io/assignment2
_
bluesky 6/10
2/11/26, 12:31 PM H
H
female = obs
_
_
F - p_
female
male = obs
_
_
M - p_
male
Assignment 2: Analyzing Bluesky | Social Algorithms
Interpretation:
H > 0 indicates homophily: same-gender repliers are over-r e pr e s en t e d relative to
baseline
H < 0 indicates heterophily: same-gender repliers are un d er-r e pr e s en t e d
H = 0 indicates no preference: replier gender matches the overall baseline
Example: If the baseline is 40% female ( p_
female = 0.40) and female senators receive 45%
female repliers ( obs
_
F = 0.45), then H
_
female = 0.45 - 0.40 = +0.05, indicating mild
homophily.
Create a visualization showing the gender breakdown of repliers for female vs. male senators. Is
there evidence of homophily?
1. Statistical significance: Assess whether the observed homophily is statistically significant or
could plausibly arise by chance. There are many ways to structure this inquiry. Explain your
approach. As a guiding principle, the null hypothesis is that replier gender is independent of
senator gender (i.e., all senators draw from the same baseline distribution)
Report whether your homophily findings are statistically significant and discuss what this
means for your conclusions.
II.4 Reply Timing Analysis
Since the API preferentially returns early replies, investigate whether early and late replies differ.
Focus on posts with 50–200 total replies (where you likely captured most replies):
1. Split replies into “early” (first 25% by timestamp) and “late” (last 25%)
2. Compare the gender composition of early vs. late replies
3. Look for other differences: reply length, engagement received (likes), or any other quantifiable
patterns you observe
This analysis helps you understand the bias introduced by the API’s ~200-reply limit and reason
about what you might be missing for high-engagement posts. What senators are often getting more
than 200 replies per post? Given what this timing analysis shows, do you have any caveats you
would add to your previous analysis? Are they minor or major caveats?
API limits frequently require analysis like this one to include caveats/hedges, so becoming
comfortable with interpreting the limitations of the data you collected is very good practice.
https://socialalgorithms.github.io/assignment2
_
bluesky 7/10
2/11/26, 12:31 PM II.5 Reflection Questions
Assignment 2: Analyzing Bluesky | Social Algorithms
Include brief responses to these questions in your report:
1. What are the key limitations of inferring gender from first names?
2. We focused on gender homophily. What other forms of homophily might be interesting to
study on Bluesky (e.g., geographic, topical, temporal), and how would you operationalize
them?
3. For a senator whose posts routinely receive 1,000+ replies, roughly what fraction of replies are
we capturing? What might be systematically different about the replies we miss? Based on
your reply timing analysis, do early replies appear different from late replies? How should
these limitations affect our confidence in the homophily estimates?
Part III: Platform biases
1. Our dataset contains 42 senators: 41 Democrats and 1 Independent. No Republican senators
have verified Bluesky accounts. This complete partisan skew raises important questions about
what we can learn from this analysis. What does this tell us about the generalizability of any
findings about “senators” or “political figures”? With zero Republicans, can we make any claims
about partisan patterns?
2. If we wanted to study partisan homophily (do Democrats engage more with Democrats?),
what alternative approaches or data sources might we consider? Briefly research senator
activity on other platforms and sketch out a plan for how you’d study partisan homophily.
3. How might the partisan composition of Bluesky’s overall user base affect the homophily
patterns we observe, even when studying only gender?
Deliverables
Submit the following:
1. Report (PDF, 6 pages max) containing:
“Senators You May Know” recommendation table and discussion
Feed overlap analysis (both heatmaps)
Homophily results with visualization
Reply timing analysis
Responses to reflection questions in Parts I, II, and III
https://socialalgorithms.github.io/assignment2
_
bluesky 8/10
2/11/26, 12:31 PM Assignment 2: Analyzing Bluesky | Social Algorithms
2. Any code you wrote, either as python scripts or notebooks
Grading Rubric
Component Weight Criteria
Data Collection 15%
Feed Analysis (Recommendations +
similarity)
20%
Homophily Analysis 20%
Reply Timing Analysis 10%
Written Questions 20%
Code & Report Quality 15% Correct API usage, rate limiting, handles
edge cases
Correct algorithm, clear table/heatmaps,
interpretation
Sound methodology, appropriate
statistics, visualization
Thoughtful comparison, addresses API
limitations
Depth of reflection, honest engagement
with limitations
Clean code, clear writing, effective figures
Technical Reference
See bluesky_
helpers.py for:
API wrapper functions: get
_profile(), get
_
follows(), get
all
_
_
follows(),
get
author
_
_
feed(), get
_post
thread()
_
Utility functions: parse
_
datetime(), is
within
_
_
hours(), load
_
senators(),
save
_json(), load
_json()
Gender inference scaffolding: load
name
data() and infer
_
_
_gender() — you must
implement these
Run python bluesky_
helpers.py to test the API functions and see example usage.
For official API documentation, see: Bluesky API Reference
https://socialalgorithms.github.io/assignment2
_
bluesky 9/10
2/11/26, 12:31 PM Academic Integrity
Assignment 2: Analyzing Bluesky | Social Algorithms
You may work in groups of up to 3 students. List all group members on your report and code files.
You may use the provided helper code and consult API documentation. Do not share code with
other groups. Describe your use of AI tools (AI is permitted).
Good luck!