# Data Model: Docusaurus Platform

## Curriculum Module
- **name**: string (e.g., "The Robotic Nervous System (ROS 2)")
- **id**: string (e.g., "module-1", "module-2")
- **description**: string
- **chapters**: array of Chapter entities
- **order**: integer (for sequencing modules)

## Chapter
- **title**: string (e.g., "Nodes/Topics/Services")
- **id**: string (e.g., "nodes-topics-services")
- **module_id**: string (reference to parent module)
- **content_path**: string (path to markdown file)
- **sub_chapters**: array of SubChapter entities
- **order**: integer (for sequencing chapters within module)

## SubChapter
- **title**: string
- **id**: string
- **chapter_id**: string (reference to parent chapter)
- **content_path**: string (path to markdown file)
- **order**: integer (for sequencing sub-chapters within chapter)

## Curriculum Navigation
- **current_module**: Module entity
- **current_chapter**: Chapter entity
- **current_sub_chapter**: SubChapter entity
- **progress**: float (0-100 percentage)
- **user_preferences**: object (for future personalization)

## Translation State (Placeholder)
- **current_language**: string ("en" default, "ur" for Urdu)
- **available_languages**: array of strings
- **translation_status**: object (status of translation for each content piece)