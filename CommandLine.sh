# Merge the course files course_*.tsv and save the result in merged_courses.tsv
cat course_*.tsv | sort -u > merged_courses.tsv

# Find the country with the highest number of master's degrees
most_common_country=$(tail -n +2 merged_courses.tsv | cut -f 10 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | sort | uniq -c | sort -nr | head -n 1 | awk '{print $2, $3}')

echo "The country with the highest number of master's degrees is: $most_common_country"

# Find the city in the country with the highest number of master's degrees
most_common_city=$(grep "$most_common_country" merged_courses.tsv | cut -f 11 | sort | uniq -c | sort -nr | head -n 1 | awk '{print $2}')

echo "The city in the country with the highest number of master's degrees is: $most_common_city"

# Count how many colleges offer part-time education
part_time_count=$(awk -F'\t' '$4 ~ /Part Time/ {count++} END {print count}' merged_courses.tsv)

echo "The number of colleges offering part-time education is: $part_time_count"

# Calculate the percentage of courses in Engineering
engineering_courses_percentage=$(grep -i 'Engineer' merged_courses.tsv | wc -l)
total_courses=$(tail -n +2 merged_courses.tsv | wc -l)

if [ "$total_courses" -gt 0 ]; then
    engineering_percentage=$((engineering_courses_percentage * 100 / total_courses))
else
    engineering_percentage=0
fi

echo "The percentage of courses in Engineering is: $engineering_percentage%"
