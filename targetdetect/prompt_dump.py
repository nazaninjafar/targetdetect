candidate_prompts = ["""Given the text highlight or underline parts of the text that mention or refer to the specific target.
Please note that sometimes in the text the target is not explicitly mentioned and you have to look for parts that implicitly refer to the target.
It is also possible that the target (explicitly) and the implicit characteristics are mentioned at the same time. So please make sure to highlight any of those references to the target.
if the text is not hate speech at all do not highlight any target. 
Input is in the following format: Input: <>. Output the extracted spans as a python list: [ <first span> , <second span> ,…, <last span>]
Examples: 
- Input:< : hate posters ? you want hate buddy ? read the koran . goo ...> 
Output: ["koran"]
- Input <"I would really prefer to have true British neighbours instead of all these curryheads"> 
Output: ["these curryheads" ]
- Input: <both north america and europe belong to us . white men conquered and colonized both .> 
Output: []
- Input: <"I dont want to mix with a barren woman as she will envy my kids"> 
Output: [ "barren woman", " she"] 
- Input: <" the biggest lie of all is the fake jewish  "" holocaust "" "" narrative of wwii ! on the importance of exposing the holohoax \">
Output: [ "jewish", "holocaust", "holohoax"]
- Input: <hey ignorant you think all whites are rednecks  have blonde hair.many latinos are white  and you're afraid of them marrying your daughter . lol> 
Output: []
""","""The task is to highlight multiple text spans from the given input hate speech content that mention, refer or explicitly and/or implicitly a specific protected group or their representation or characteristics that has been targeted. If there is no hate speech, the output will be empty. Input is in the following format: Input: <>. Output the extracted spans as a Python list: [ <first span> , <second span> ,…, <last span>].
Examples:
- Input:< : hate posters ? you want hate buddy ? read the koran . goo ...> 
Output: ["koran"] 
- Input <"I would really prefer to have true British neighbours instead of all these curryheads"> 
Output: ["these curryheads"]  
- Input: <both north america and europe belong to us . white men conquered and colonized both .> 
Output: []
- Input: <"I dont want to mix with a barren woman as she will envy my kids"> 
Output: ["barren women"] 
- Input: <" the biggest lie of all is the fake jewish  "" holocaust "" "" narrative of wwii ! on the importance of exposing the holohoax \""">
Output: ["jewish holocaust"]
- Input: <hey ignorant you think all whites are rednecks  have blonde hair.many latinos are white  and you're afraid of them marrying your daughter . lol> 
Output: ["latinos"] 
"""]


archives = [""" The task is to highlight the part of the given text that is referring to a specific protected group that has been targeted in the content. You will be highlighting the words or phrases that directly mention the target and/or the representation (characteristics) of the target. 
Input is in the following format: Input: <>. Output the extracted spans as a python list: [ <first span> , <second span> ,…, <last span>]
Examples: 
- Input:< : hate posters ? you want hate buddy ? read the koran . goo ...> 
Output: ["koran"] 
- Input <"I would really prefer to have true British neighbours instead of all these curryheads"> 
Output: ["curryheads"]
- Input: <both north america and europe belong to us . white men conquered and colonized both .> 
Output: [" white men"] 
- Input: <"I dont want to mix with a barren woman as she will envy my kids"> 
Output: ["barren women", "she"]
- Input: <" the biggest lie of all is the fake jewish  "" holocaust "" "" narrative of wwii ! on the importance of exposing the holohoax \">
Output: ["jewish", " "" holocaust "" \""", "holohoax"]
- Input: <hey ignorant you think all whites are rednecks  have blonde hair.many latinos are white  and you're afraid of them marrying your daughter . lol> 
Output: ["latinos"] """, """The task is to identify and highlight the person or group that is a target of hate speech in the input sentence and also the qualities of them that are hated. If there is no hate speech, the output will be empty. Input is in the following format: Input: <>. Output the extracted spans as a Python list: [ <first span> , <second span> ,…, <last span>].
Examples:
- Input:< : hate posters ? you want hate buddy ? read the koran . goo ...> 
Output:  ["koran"] 
- Input <"I would really prefer to have true British neighbours instead of all these curryheads"> 
Output:  ["these curryheads"]
- Input: <both north america and europe belong to us . white men conquered and colonized both .> 
Output: []
- Input: <"I dont want to mix with a barren woman as she will envy my kids"> 
Output: ["barren woman"]
- Input: <" the biggest lie of all is the fake jewish  "" holocaust "" "" narrative of wwii ! on the importance of exposing the holohoax \""">
Output: ["jewish", "holocaust"] 
- Input: <hey ignorant you think all whites are rednecks  have blonde hair.many latinos are white  and you're afraid of them marrying your daughter . lol> 
Output: []""","""The task is to identify the object of the speech. That is, who is the speaker talking about? Include pronominal references to the object, even if the object is also explicitly listed. Ignore whether the speech is harmful to that object. Input is in the following format: Input: <>.  Output the extracted spans as a Python list: [ <first span> , <second span> ,…, <last span>].
Examples: 
- Input:< : hate posters ? you want hate buddy ? read the koran . goo ...> 
Output: ["you", "koran"] 
- Input <"I would really prefer to have true British neighbours instead of all these curryheads"> 
Output: ["these curryheads"] 
- Input: <both north america and europe belong to us . white men conquered and colonized both .> 
Output: [ "white men"]
- Input: <"I dont want to mix with a barren woman as she will envy my kids"> 
Output: ["barren women" , "she"] 
- Input: <" the biggest lie of all is the fake jewish  "" holocaust "" "" narrative of wwii ! on the importance of exposing the holohoax \""">
Output: ["jewish holocaust"]
- Input: <hey ignorant you think all whites are rednecks  have blonde hair.many latinos are white  and you're afraid of them marrying your daughter . lol> 
Output: ["many latinos"] 
"""]





