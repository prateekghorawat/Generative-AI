#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from project.crew import Project

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = {
        'motion': 'AI need strict laws for Business Optimization',
    }
    
    try:
        result = Project().crew().kickoff(inputs=inputs)
        print(result.raw)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")