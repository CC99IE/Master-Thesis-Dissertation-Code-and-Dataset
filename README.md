# Master-Thesis-Dissertation-Code-and-Dataset
The code for my master's thesis presentation and the dataset

The repo contains Main.py, the main orchestration for the analysis pipeline. This is where you also add the wanted file to test, in the global parameters, followed by 
      modules/
          -ingestion.py - file traversal, and sanity checks
          -authenticity.py - initial authenticity analysis
          -metadata.py - extraction and analysis
          -content_analysis.py - content and context analysis
          -similarity.py - checks for similarity
          -ml_classifier.py - machine learning working on analysis of real-vs-fake content and anomalies
          -charts.py - charts
          -reporting.py - accumulation of all reports, their aggregation and creation of the final report
          
  
BE AWARE: The Dataset includes viruses that may have been present on Afghanistan's governmental computers at the time of capture, as well as personal data and information. The beloved word that you are looking for to open them is, of course, "infected"

Afganistan Leaks - https://1drv.ms/u/c/7589d66e4d4d116e/EQ5DY09yJn1IoXzFJV1LAdkBLBlmEwwt4vkR3Sxa47kCXA?e=q3f5rw
