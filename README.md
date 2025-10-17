# contextual-personal-assistant-with-langchain-agent

# 🧠 Contextual Personal Assistant

An intelligent personal assistant that uses **LangChain agents** and **OpenAI GPT-4o-mini** to automatically organize, categorize, and analyze your notes, tasks, and ideas. Built with FastAPI backend and Streamlit frontend.

## ✨ Features

### 🤖 Dual AI Agents
- **Ingestion Agent**: Enhanced entity extraction using LangChain + GPT-4o-mini
- **Thinking Agent**: AI-powered insights generation with natural language summaries

### 📝 Smart Note Processing
- **Automatic Classification**: Tasks, Reminders, Ideas/Notes
- **Entity Extraction**: Dates, times, assignees, keywords
- **Semantic Organization**: Auto-categorization into topic-based envelopes
- **Context Building**: Dynamic user profile from all notes

### 🔍 Intelligent Analysis
- **Scheduling Conflicts**: Detect overlapping tasks
- **Duplicate Detection**: Find similar tasks across envelopes
- **Priority Tasks**: Identify upcoming deadlines
- **Overdue Tracking**: Monitor missed deadlines
- **Envelope Merging**: Suggest topic consolidation
- **Next Steps**: AI-generated recommendations

### 🎯 User Experience
- **Modern Web UI**: Clean Streamlit interface
- **Real-time Processing**: Instant note analysis
- **Visual Insights**: Progress bars, metrics, and summaries
- **Data Management**: Delete cards and envelopes
- **Health Monitoring**: System status and agent health

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd contextual-personal-assistant
   ```

2. **Set up environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

4. **Configure API key**
   ```bash
   # Create .env file in backend directory
   echo "OPENAI_API_KEY=sk-your-key-here" > .env
   ```

5. **Start the backend**
   ```bash
   cd backend
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Start the frontend** (in a new terminal)
   ```bash
   cd frontend
   pip install streamlit requests
   streamlit run streamlit_ui.py
   ```

7. **Access the application**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## 📖 Usage Examples

### Adding Notes
Simply type natural language notes and the AI will automatically:

```
"Call Sarah about the Q3 budget next Monday"
→ Task: Call Sarah about Q3 budget
→ Date: 2024-01-15 (next Monday)
→ Assignee: Sarah
→ Envelope: Budget & Sarah

"Idea: new logo should be blue and green"
→ Idea/Note: new logo should be blue and green
→ Keywords: logo, blue, green
→ Envelope: Design & Logo

"Remember to pick up milk on the way home"
→ Reminder: pick up milk
→ Keywords: milk, home
→ Envelope: Shopping & Milk
```

### AI Insights
The Thinking Agent analyzes all your data and provides:

- **⚡ Priority Tasks**: Upcoming deadlines (next 3 days)
- **🔴 Overdue Tasks**: Missed deadlines with days overdue
- **⚠️ Scheduling Conflicts**: Multiple tasks for same person/date
- **🔄 Duplicate Tasks**: Similar tasks across envelopes
- **🔗 Merge Suggestions**: Envelopes with overlapping topics
- **💡 Next Steps**: AI recommendations for project completion

## 🏗️ Architecture & Design

### Agent Development Framework Choice: **LangChain**
**Justification**: LangChain was selected as the primary agent development framework because:
- **Modular Design**: Clean separation between ingestion and thinking agents
- **LLM Integration**: Seamless OpenAI GPT-4o-mini integration with structured prompts
- **Error Handling**: Built-in fallback mechanisms when LLM calls fail
- **Prompt Management**: Template-based prompts ensure consistent AI responses
- **Extensibility**: Easy to add new agents or modify existing logic

### Storage Mechanism: **SQLite**
**Justification**: SQLite was chosen for local storage because:
- **Zero Configuration**: No external database server required
- **Embedding Support**: Native BLOB storage for vector embeddings
- **Performance**: Sufficient for single-user personal assistant use case
- **Portability**: Single file database, easy to backup and transfer
- **ACID Compliance**: Reliable data integrity for user's personal data

### System Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT (Raw Note)                     │
│         "Call Sarah about Q3 budget next Monday"             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         INGESTION & ORGANIZATION AGENT (LangChain)          │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  1. LLM Entity Extraction                           │   │
│  │     • Card Type Classification (Task/Reminder/Note) │   │
│  │     • Date/Time Parsing (with context awareness)    │   │
│  │     • Assignee Detection (NER via LLM)              │   │
│  │     • Keyword Extraction (semantic understanding)   │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  2. Envelope Assignment (LLM + Embeddings)          │   │
│  │     • Semantic matching with existing envelopes     │   │
│  │     • Context-aware grouping decisions              │   │
│  │     • Auto-create new envelopes when needed         │   │
│  └─────────────────────────────────────────────────────┘   │
│                     │                                         │
│                     ▼                                         │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  3. Context Update                                 │   │
│  │     • Update user's active projects                 │   │
│  │     • Track contacts and deadlines                  │   │
│  │     • Refine themes and keywords                    │   │
│  └─────────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    STRUCTURED OUTPUT                          │
│                                                               │
│  📝 Card: {                                                  │
│    type: "Task",                                             │
│    description: "Call Sarah about Q3 budget",                │
│    date: "2024-01-15",                                       │
│    assignee: "Sarah",                                         │
│    context_keywords: ["budget", "Q3", "call"]                │
│  }                                                           │
│                                                               │
│  📦 Envelope: "Budget & Sarah" (ID: 3)                       │
│                                                               │
│  👤 Context Updates: {                                       │
│    active_projects: ["Budget & Sarah"],                       │
│    contacts: ["Sarah"],                                      │
│    upcoming_deadlines: ["2024-01-15"]                        │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

### Thinking Agent Design

The Thinking Agent operates on a scheduled basis (triggered via `/think` endpoint) and analyzes all existing data:

**Logic Flow**:
1. **Data Collection**: Gather all Cards, Envelopes, and User Context
2. **Pattern Analysis**: Cross-reference data for patterns and relationships
3. **Conflict Detection**: Identify scheduling conflicts and overlaps
4. **Insight Generation**: Use LLM to generate natural language summaries
5. **Recommendation Engine**: Suggest next steps and optimizations

**Analysis Categories**:
- **Scheduling Conflicts**: Multiple tasks for same person/date
- **Duplicate Detection**: Similar tasks across different envelopes
- **Priority Tasks**: Upcoming deadlines (next 3 days)
- **Overdue Items**: Missed deadlines with severity tracking
- **Envelope Merging**: Suggest consolidation of related topics
- **Next Steps**: AI-generated recommendations for project completion

### Backend (FastAPI)
- **`app.py`**: Main FastAPI application with dual LangChain agents
- **Database**: SQLite with embeddings for semantic search
- **Models**: spaCy for NLP, SentenceTransformers for embeddings
- **Agents**: LangChain + OpenAI GPT-4o-mini for enhanced processing

### Frontend (Streamlit)
- **`streamlit_ui.py`**: Modern web interface
- **Real-time Updates**: Live data refresh and processing
- **Visual Analytics**: Progress bars, metrics, and insights
- **Responsive Design**: Clean, intuitive user experience

## 🔧 API Endpoints

### Core Operations
- `POST /add` - Process new note with AI
- `GET /cards` - List all cards (with optional envelope filter)
- `GET /envelopes` - List all envelopes with card counts
- `GET /context` - Get user context (projects, contacts, themes)
- `GET /think` - Run AI analysis and generate insights

### Management
- `DELETE /cards/{id}` - Delete specific card
- `DELETE /envelopes/{id}` - Delete envelope (moves cards to Miscellaneous)
- `POST /envelopes/cleanup` - Merge duplicate Miscellaneous envelopes

### System
- `GET /health` - System health and agent status
- `GET /think/debug` - Raw insights data without AI formatting

## 🛠️ Configuration

### Environment Variables
```bash
OPENAI_API_KEY=sk-your-key-here  # Required for AI agents
```

### Model Selection & Justification

**LLM Model: OpenAI GPT-4o-mini**
- **Reasoning**: Chosen for superior entity extraction and classification accuracy
- **Cost Consideration**: ~$0.15/1M input tokens, ~$0.60/1M output tokens
- **Scalability**: API-based, scales automatically with usage
- **Performance**: Excellent at understanding context and extracting structured data
- **Alternative**: Could use local models (Llama, Mistral) but would require significant infrastructure

**Embedding Model: all-MiniLM-L6-v2**
- **Reasoning**: Lightweight, fast, and effective for semantic similarity
- **Size**: 22MB, runs locally without API costs
- **Performance**: 384 dimensions, good balance of accuracy and speed
- **Alternative**: Could use OpenAI embeddings but adds API costs

**NLP Model: spaCy en_core_web_sm**
- **Reasoning**: Reliable baseline NER and linguistic processing
- **Performance**: Fast local processing, good entity recognition
- **Fallback**: Used when LLM extraction fails
- **Alternative**: Could use larger spaCy models for better accuracy

### Model Configuration
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)
- **LLM Model**: `gpt-4o-mini` (via OpenAI API)
- **Similarity Threshold**: 0.50 for envelope matching
- **spaCy Model**: `en_core_web_sm` for NLP processing

## 📊 Data Models

### Card Types
- **Task**: Action items with verbs (call, email, schedule)
- **Reminder**: Time-sensitive items or explicit reminders
- **Idea/Note**: General thoughts and concepts

### Entity Extraction
- **Dates**: Absolute (2024-01-15) and relative (tomorrow, next Monday)
- **Times**: 12-hour format with AM/PM support
- **Assignees**: People and organizations from NER
- **Keywords**: Nouns and proper nouns (filtered)

### Envelope Organization
- **Semantic Similarity**: Cosine similarity of embeddings
- **Keyword Overlap**: Jaccard similarity of keyword sets
- **AI Suggestions**: LangChain agent recommendations
- **Auto-naming**: Generated from top keywords

## 🔍 Advanced Features

### LangChain Integration
- **Prompt Templates**: Structured prompts for consistent AI responses
- **Output Parsers**: JSON extraction from AI responses
- **Error Handling**: Graceful fallback to base extraction
- **Temperature Control**: Optimized for extraction (0.1) and insights (0.2)

### Semantic Search
- **Vector Embeddings**: 384-dimensional sentence embeddings
- **Similarity Matching**: Cosine similarity for envelope assignment
- **Keyword Enhancement**: Combined semantic + keyword scoring

### Context Awareness
- **Dynamic Profile**: Built from all processed notes
- **Project Tracking**: Active projects from envelope names
- **Contact Management**: People mentioned in tasks
- **Deadline Monitoring**: Upcoming dates and overdue items

## 🚨 Troubleshooting

### Common Issues

1. **"Cannot connect to backend"**
   - Ensure backend is running: `uvicorn app:app --reload`
   - Check port 8000 is available

2. **"OpenAI API key not configured"**
   - Create `.env` file in backend directory
   - Add: `OPENAI_API_KEY=sk-your-key-here`

3. **"spaCy model not found"**
   - Run: `python -m spacy download en_core_web_sm`

4. **"Agent initialization failed"**
   - Check OpenAI API key is valid
   - Verify internet connection
   - System falls back to base extraction

### Performance Tips
- **Batch Processing**: Add multiple notes at once
- **Regular Cleanup**: Use envelope cleanup endpoint
- **Monitor Health**: Check `/health` endpoint regularly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is built for educational purposes as part of a Machine Learning Engineer assignment.

## 🙏 Acknowledgments

- **LangChain**: Agent framework and prompt management
- **OpenAI**: GPT-4o-mini for intelligent processing
- **spaCy**: Natural language processing
- **SentenceTransformers**: Semantic embeddings
- **FastAPI**: Modern Python web framework
- **Streamlit**: Rapid web app development

---

**Built with ❤️ using LangChain, OpenAI GPT-4o-mini, and modern Python frameworks**