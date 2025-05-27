# Task Generator – UI & API Specification

This document describes **how the React mock‑up works** (structure, component interaction, state) and the **HTTP/WS endpoints** the platform must expose to make the UI fully functional.

---

## 1  High‑level Architecture

| Layer                                           | Responsibility                                                                          |
| ----------------------------------------------- | --------------------------------------------------------------------------------------- |
| **Frontend (React + Tailwind + Framer Motion)** | Collect user input, present Kafka topics & tools, assemble a *Task Generation Request*. |
| **Backend REST API**                            | Persist templates, topics, tools, tasks; stream code generation results.                |
| **Kafka/Infra**                                 | (Runtime) Executes the generated tasks; not covered here.                               |

---

## 2  UI Component Breakdown

### 2.1  Side Bars – *Consumes* & *Produces*

- **Purpose:** let users select **one** consume topic and **N** produce topics.
- **Interaction**
  1. On mount, fetch topic list → render as scrollable buttons.
  2. Click toggles selection (single vs multi).
- **State Hooks**
  ```ts
  const [consumeTopic, setConsumeTopic] = useState<string|null>();
  const [produceTopics, setProduceTopics] = useState<string[]>();
  ```
- **API**: `GET /api/topics` (see §3.2)

### 2.2  Center Pane – *Task Generator*

| Section                | Details                                                                    |
| ---------------------- | -------------------------------------------------------------------------- |
| **Template Selector**  | `<Select>` bound to `selectedTemplate`.                                    |
| **Description Box**    | `<textarea>` free‑form task description.                                   |
| **Runtime Properties** | Key‑value creator (adds to `runtimeProps`).                                |
| **Tools Grid**         | Shows selected tools in responsive cards. Add button opens **Tool Modal**. |
| **Generate Button**    | Sends assembled payload to `POST /api/tasks` and awaits code URL / status. |

### 2.3  Tool Modal

- Left: list of available tools (`GET /api/tools`) – single‑select.
- Right: dynamic config form:
  - “Search Documentation” ⇒ checkbox multi‑select.
  - All other tools ⇒ read‑only keys + editable values.
- *Add* pushes configured tool into local `toolInstances`.

---

## 3  Required API Endpoints

### 3.1  `GET /api/templates`

Return available task templates.

```json
200 OK
[
  {"id":"scheduled","name":"Scheduled Job"},
  {"id":"message-consumer","name":"Message Consumer"}
]
```

### 3.2  `GET /api/topics`

List Kafka topics the user may reference.

```json
200 OK
[
  {"id":"orders","description":"Kafka topic for order events"},
  ...
]
```

Optional query params:

- `type=consume|produce` for future filtering.

### 3.3  `GET /api/tools`

Catalog of pluggable helper tools.

```json
[
  {"id":"schema-inference","name":"Schema Inference","configSchema":[]},
  {"id":"search-documentation","name":"Search Documentation","configSchema":[{"key":"documents","type":"string[]"}]},
  ...
]
```

`configSchema` drives the modal’s dynamic form.

### 3.4  `POST /api/tools/validate`

Optional server‑side validation of a configured tool instance.

```json
{
  "id":"test-publish-topic",
  "config": {"prefix":"dev-"}
}
```

Returns `200` or `422` with error field hints.

### 3.5  `POST /api/tasks`

Create a code‑generation task.

```json
{
  "templateId":"message-consumer",
  "description":"Sync invoices to S3",
  "consumeTopic":"invoices",
  "produceTopics":["invoices-arch"],
  "runtimeProperties":{"timeout":"30s"},
  "tools":[
    {"id":"schema-inference","config":{}},
    {"id":"s3-client","config":{"accessKeyId":"…","secretAccessKey":"…"}}
  ]
}
```

Response:

```json
202 Accepted
{"taskId":"abc123"}
```

### 3.6  `GET /api/tasks/{taskId}`

Poll task status & code artefact URL.

```json
{
  "status":"completed",
  "downloadUrl":"https://…/task-abc123.zip"
}
```

Streaming alt: `GET /api/tasks/{taskId}/events` (SSE/WebSocket).

---

## 4  Validation Rules (Frontend ⇄ Backend)

| Field             | Rule                                              |
| ----------------- | ------------------------------------------------- |
| **Description**   | 1–1 000 chars, Markdown allowed.                  |
| **Runtime Props** | Keys snake\_case, ≤ 32 chars; values ≤ 256 chars. |
| **Topic IDs**     | Must exist in `/api/topics`.                      |
| **Tools**         | Config must satisfy `configSchema`.               |

---

## 5  Future Enhancements

- **Preview Pane / Message Hashes** – not in current rollback but API stub would be `GET /api/topics/{id}/messages?limit=…`. Desirable
- **Auth** via JWT, all endpoints under `/api/v1`. Out of scope
- **RBAC** on topics (consume vs produce permissions). Out of scope

---

## 6  Open Questions (Closed)

1. Should generated code be returned inline (SSE) or via downloadable artefact? SSE
2. Is multi‑template composition in scope (e.g. Scheduled + Consumer combo)? No
3. Versioning strategy for tools & templates. Out of scope

## 7   Additional Constraints

1. The programming language for the task implementation is python.
2. There will be three generated artifacts:
   - A python script that implements the task logic.
   - A dockerfile to build the task container.
   - A JSON file containing task metadata.

---

\### End of Document

