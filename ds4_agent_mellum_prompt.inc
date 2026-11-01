/* Mellum 2.1, included by ds4_agent.c where this code sits: not a separate translation unit. */

/* Mellum 2.1's tool block, as chat_template.jinja writes it for Pi's four tools (read, bash, edit,
 * write; Pi is MIT).  Each line is the bytes `tools | tojson` emits under transformers: Python's
 * ", " and ": " separators, insertion order.  tests/ds4_agent_test.c compares the block to the
 * independent copy in tests/mellum_tool_text.h, which tests/test_mellum_config.c in turn compares
 * to the jinja2 render, so a typo here is a red test. */
static const char *const agent_hermes_tool_json[4] = {
    "{\"type\": \"function\", \"function\": {\"name\": \"read\", \"description\": \"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.\""
    ", \"parameters\": {\"type\": \"object\", \"required\": [\"path\"], \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Path to the file to read (relative or absolute)\"}, \"offset\": {\"type\": \"number\", \"description\": \"Line number to start reading from (1-indexed)\"}, \"limit\": {\"type\": \"number\", \"description\": \"Maximum number of lines to read\"}}}, \"strict\": false}}",
    "{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.\""
    ", \"parameters\": {\"type\": \"object\", \"required\": [\"command\"], \"properties\": {\"command\": {\"type\": \"string\", \"description\": \"Shell command to execute\"}, \"timeout\": {\"type\": \"number\", \"description\": \"Timeout in seconds (optional, no default timeout)\"}}}, \"strict\": false}}",
    "{\"type\": \"function\", \"function\": {\"name\": \"edit\", \"description\": \"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.\""
    ", \"parameters\": {\"type\": \"object\", \"required\": [\"path\", \"edits\"], \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Path to the file to edit (relative or absolute)\"}, \"edits\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"required\": [\"oldText\", \"newText\"], \"properties\": {\"oldText\": {\"type\": \"string\", \"description\": \"Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.\"}, \"newText\": {\"type\": \"string\", \"description\": \"Replacement text for this targeted edit.\"}}}, \"description\": \"One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.\"}}}, \"strict\": false}}",
    "{\"type\": \"function\", \"function\": {\"name\": \"write\", \"description\": \"Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.\""
    ", \"parameters\": {\"type\": \"object\", \"required\": [\"path\", \"content\"], \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Path to the file to write (relative or absolute)\"}, \"content\": {\"type\": \"string\", \"description\": \"Content to write to the file\"}}}, \"strict\": false}}",
};

static const char agent_hermes_tools_head[] =
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
static const char agent_hermes_tools_tail[] =
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

static char *agent_build_hermes_tools_block(void) {
    size_t cap = strlen(agent_hermes_tools_head) + strlen(agent_hermes_tools_tail) + 1;
    for (int i = 0; i < 4; i++) cap += 1 + strlen(agent_hermes_tool_json[i]);
    char *out = xmalloc(cap), *q = out;
    q += sprintf(q, "%s", agent_hermes_tools_head);
    for (int i = 0; i < 4; i++) q += sprintf(q, "\n%s", agent_hermes_tool_json[i]);
    sprintf(q, "%s", agent_hermes_tools_tail);
    return out;
}

/* The one system turn's content: the preamble (when there is one), a blank line, the tools block.
 * The caller renders it as a single system message; the template has no second one before the
 * first user turn. */
static char *agent_build_hermes_system_text(const char *preamble) {
    char *block = agent_build_hermes_tools_block();
    if (!preamble) return block;
    size_t cap = strlen(preamble) + 2 + strlen(block) + 1;
    char *out = xmalloc(cap);
    snprintf(out, cap, "%s\n\n%s", preamble, block);
    free(block);
    return out;
}

/* What the agent says ahead of the tools block: a short Pi-flavoured intro, the user's -sys text,
 * the session's date and time, the working directory. */
static char *agent_build_hermes_preamble(const char *extra, const char *datetime, const char *cwd) {
    static const char intro[] =
        "You are a coding agent running in a local workspace. Use the tools to read, write and edit "
        "files and to run commands.\n\n"
        "Guidelines:\n"
        "- Use bash for file operations like ls, rg and find.\n"
        "- Be concise in your responses.\n"
        "- Show file paths clearly when working with files.";
    size_t cap = sizeof(intro) + 4;
    if (extra && extra[0]) cap += strlen(extra) + 2;
    if (datetime && datetime[0]) cap += strlen(datetime) + 2;
    if (cwd && cwd[0]) cap += strlen(cwd) + 32;
    char *out = xmalloc(cap), *q = out;
    q += sprintf(q, "%s", intro);
    if (extra && extra[0]) q += sprintf(q, "\n\n%s", extra);
    if (datetime && datetime[0]) q += sprintf(q, "\n\n%s", datetime);
    if (cwd && cwd[0]) q += sprintf(q, "\n\nCurrent working directory: %s", cwd);
    return out;
}
