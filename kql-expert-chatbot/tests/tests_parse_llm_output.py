import unittest
from my_productivity_tool import parse_llm_response

class TestParseLLMOutput(unittest.TestCase):

    def test_response_simple(self):
        output = '''
summary: Updated Objective and Instructions, added relevant code

[Objective]
Find the last 5 screenshots taken

[Instructions]
1. Identify the folder containing screenshots
2. List the files in the folder
3. Filter the results to only show screenshots
4. Sort the screenshots by date
5. Display the last 5 screenshots

[Code]
#!/bin/bash

SCREENSHOTS_FOLDER="$HOME/Pictures/Screenshots"
FILE_PATTERN="*.png"

cd "$SCREENSHOTS_FOLDER"

ls -lt $FILE_PATTERN | head -n 5
        '''

        s, o, i, c = parse_llm_response(output)

        print("Summary:")
        print(s)
        print("\nObjective:")
        print(o)
        print("\nInstructions:")
        print(i)
        print("\nCode:")
        print(c)

        print("#####################")

    def test_response_with_back_ticks(self):
        output='''
summary: Updated objective, added instructions, and generated code

[Objective]
Find the last 5 screenshots taken by the user

[Instructions]
1. Locate the folder where screenshots are saved
2. List the files in the folder
3. Filter the list to include only screenshots
4. Sort the list by date, with the most recent screenshots first
5. Display the last 5 screenshots in the list

[Code]
```bash
#!/bin/bash
screenshot_folder="$HOME/Pictures/Screenshots"
screenshot_files=$(find "$screenshot_folder" -type f -name "*.png" -o -name "*.jpg" -o -name "*.jpeg")
sorted_files=$(echo "$screenshot_files" | xargs -I {} stat --format="%Y %n" {} | sort -nr | cut -d ' ' -f 2-)
last_five_screenshots=$(echo "$sorted_files" | head -n 5)
echo "Last 5 screenshots:"
echo "$last_five_screenshots"
        '''

        s, o, i, c = parse_llm_response(output)

        print("Summary:")
        print(s)
        print("\nObjective:")
        print(o)
        print("\nInstructions:")
        print(i)
        print("\nCode:")
        print(c)

        print("#####################")

    def test_response3(self):
        output='''
summary: Updated Instructions to include the location of repositories

[Objective]
Build a tool that will scan my repositories and pull out all TODO messages.

[Instructions]
1. Navigate to the root directory of the repository at ~/Code.
2. Search for all files in the repository.
3. For each file, search for lines containing 'TODO' messages.
4. Print the file path and the TODO messages.

[Code]
#!/bin/bash
cd ~/Code
find . -type f -print0 | xargs -0 grep -n "TODO" --color=always
        '''

        s, o, i, c = parse_llm_response(output)

        print("Summary:")
        print(s)
        print("\nObjective:")
        print(o)
        print("\nInstructions:")
        print(i)
        print("\nCode:")
        print(c)

        print("#####################")
    
    def test_response4(self):
        output='''
                summary: Updated Instructions to limit TODOs to 3 per repository.

        [Objective]
        Build a tool that will scan my repositories and pull out all TODO messages.

        [Instructions]
        1. Navigate to the root directory of the repository at ~/Code.
        2. Search for all files in the repository.
        3. For each file, search for lines containing 'TODO' messages.
        4. Limit the output to 3 TODO messages per repository.
        5. If there are more than 3 TODO messages, print "and X more" where X is the remaining number of TODO messages.
        6. Print the file path and the TODO messages.

        [Code]
        #!/bin/bash
        cd ~/Code
        for repo in */; do
          echo "Repository: $repo"
          todo_count=0
          total_count=0
          while IFS= read -r line; do
            if [[ $line == *"TODO"* ]]; then
              total_count=$((total_count + 1))
              if [[ $todo_count -lt 3 ]]; then
                echo "$line"
                todo_count=$((todo_count + 1))
              fi
            fi
          done < <(grep -r "TODO" "$repo")
          remaining=$((total_count - todo_count))
          if [[ $remaining -gt 0 ]]; then
            echo "and $remaining more"
          fi
          echo ""
        done
        '''

        s, o, i, c = parse_llm_response(output)

        print("Summary:")
        print(s)
        print("\nObjective:")
        print(o)
        print("\nInstructions:")
        print(i)
        print("\nCode:")
        print(c)

        print("#####################")