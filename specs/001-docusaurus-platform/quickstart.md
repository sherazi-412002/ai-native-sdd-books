# Quickstart: Docusaurus Platform Development

## Prerequisites
- Node.js 18+ installed
- npm or yarn package manager
- Git for version control

## Setup Instructions

1. **Install Dependencies**
   ```bash
   cd website
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm start
   ```
   This will start the Docusaurus development server at http://localhost:3000

3. **Build for Production**
   ```bash
   npm run build
   ```

4. **Serve Production Build Locally**
   ```bash
   npm run serve
   ```

## Adding New Content

1. **Create new module**: Add a new directory under `docs/` with `_category_.json`
2. **Add chapters**: Create markdown files within the module directory
3. **Update navigation**: The sidebar will automatically update based on folder structure

## Development Guidelines

- All components must use TypeScript
- Styling must use CSS Modules for the landing page
- Follow the three-tier hierarchy: Modules -> Chapters -> Sub-chapters
- Include translation placeholder on all content pages
- Maintain the futuristic design aesthetic