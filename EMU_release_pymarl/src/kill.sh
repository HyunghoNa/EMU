#!/bin/bash

ps aux | grep python | awk '{print $2}' | xargs kill
ps aux | grep SC2 | awk '{print $2}' | xargs kill
