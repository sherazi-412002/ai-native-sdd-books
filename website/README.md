# The Autonomous Humanoid: A Comprehensive Guide

This is a [Docusaurus](https://docusaurus.io/) project for the Autonomous Humanoid curriculum, featuring a structured 4-module robotics curriculum covering ROS 2, Digital Twins, AI-Robotics, and Vision-Language-Action concepts.

## Installation

```bash
npm install
```

## Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

Using SSH:

```bash
USE_SSH=true npm run deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> npm run deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
