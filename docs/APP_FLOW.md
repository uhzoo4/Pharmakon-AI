# Pharmakon – Application Flow

This document details the interactive flow of the Pharmakon system, showing the sequences of requests, states, and client-server processing loops.

---

## 1. Sequence Diagram: Text Generation (SSE)

This diagram tracks the lifecycle of a text generation request, showing the streaming mechanics over Server-Sent Events (SSE).

```mermaid
sequenceDiagram
    autonumber
    actor User as Writer
    participant UI as Next.js Client
    participant API as FastAPI Backend
    participant Model as NumPy Engine

    User->>UI: Types prompt & clicks Send
    Note over UI: Disable input, show typing card
    UI->>API: HTTP POST /api/generate (Payload with configuration)
    API->>API: Load Active Personality Weights
    API->>Model: Primary Tokenization (text -> indices)
    
    loop Inference Stream
        API->>Model: Forward Pass (causal attention, RoPE)
        Model-->>API: Logits (seq_len, vocab_size)
        API->>API: Sampler (temperature, blacklist)
        API->>API: Decode Index (index -> char)
        API-->>UI: EventSource message Yield: {"text": char}
        UI->>UI: Append character to card (typewriter animation)
    end

    API-->>UI: EventSource message Yield: {"done": true}
    Note over UI: Re-enable input, finalize text wrapper
```

---

## 2. Sequence Diagram: Personality Swapping

This diagram shows how weights are swapped in memory without latency when a user changes their active personality in the UI.

```mermaid
sequenceDiagram
    autonumber
    actor User as Writer
    participant UI as Next.js Client
    participant API as FastAPI Backend
    participant Manager as WeightManager

    User->>UI: Clicks book cover (e.g., Kafkaesque)
    UI->>UI: Slide transition & theme color swap
    UI->>API: HTTP POST /api/generate (request using "kafkaesque")
    API->>Manager: get_weights("kafkaesque")
    Note over Manager: Retrieve array references from RAM cache
    Manager-->>API: Dictionary of parameter matrices
    API->>API: Bind parameters to model references
    Note over API: Pointer swap completed (< 1ms)
```

---

## 3. Application State Transitions

```mermaid
stateDiagram-v2
    [*] --> Landing : Launch App
    
    state Landing {
        [*] --> CarouselActive
        CarouselActive --> CarouselActive : Hover / Rotate Cover
        CarouselActive --> SelectPersonality : Click Personality
    }
    
    SelectPersonality --> ChatActive : Initialize Theme & Sounds
    
    state ChatActive {
        [*] --> InputIdle
        InputIdle --> Generating : Type Prompt & Press Enter
        Generating --> Generating : Stream Characters
        Generating --> InputIdle : End Token or Terminate Connection
        InputIdle --> SettingsOpen : Toggle Cog Icon
        SettingsOpen --> InputIdle : Close Settings Drawer
    }
    
    ChatActive --> Landing : Reset / Click Title
```
