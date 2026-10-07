# RASCAL — Android application

Kotlin Android implementation of RASCAL for field exploration on phones and tablets.

## Contents

- **app/** — Android application module
  - `src/main/java/` — Kotlin source code
  - `src/main/res/` — layouts, drawables, icons and other resources
  - `build.gradle` — module-level Gradle build file
- **build.gradle** — top-level Gradle build file
- **settings.gradle** — project settings
- **gradle.properties** — Gradle configuration
- **gradlew** / **gradlew.bat** — Gradle wrapper scripts
- **gradle/** — Gradle wrapper files

## Requirements

- Android Studio (recommended) or a command-line Gradle setup
- Android SDK with API level 35 installed
- JDK 17 or later

## Build from source

### With Android Studio

1. Open the `android/` folder in Android Studio.
2. Let Gradle sync finish.
3. Choose **Build → Build Bundle(s) / APK(s) → Build APK(s)**.

### From the command line

```bash
./gradlew assembleRelease
```

On Windows:

```powershell
.\gradlew.bat assembleRelease
```

The signed release APK will be produced in:

```text
app/build/outputs/apk/release/app-release-unsigned.apk
```

For a debug build:

```bash
./gradlew assembleDebug
```

## Install the pre-built APK

A ready-to-install APK is available in [`download/android/`](../download/android/).

## Documentation

The Android user manual is available in [`documentation/android/`](../documentation/android/).
