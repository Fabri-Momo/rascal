# RASCAL — ImageJ/Fiji plugin

Java implementation of RASCAL as an ImageJ/Fiji plugin.

## Contents

- **src/main/java/Rascal_GL.java** — Java source code
- **pom.xml** — Maven project file

## Requirements

- ImageJ or Fiji
- Java 8 or later (tested with Java 17.0.2)
- Maven 3.9.16 (for building from source)

## Building from source

### Plugin JAR for Fiji

```bash
mvn clean package
```

This produces `target/Rascal_GL.jar`, a lightweight plugin that uses dependencies provided by **Fiji** at runtime.

### Standalone fat JAR for ImageJ

```bash
mvn clean package -P imagej
```

This produces `target/Rascal_GL-standalone.jar`, a self-contained JAR with all dependencies bundled. Use this version with a plain **ImageJ** installation.

## Installation

Copy the `.jar` matching your environment to your ImageJ or Fiji `plugins/` folder and restart the application.

## Pre-built plugin

Pre-built JARs are available in the [`download/imagej/`](../download/imagej/) directory.
