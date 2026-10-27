/* Mellum 2.1's tool-bearing chat text, written the way chat_template.jinja writes it.
 * Header-only so tests/test_mellum_config.c and tests/ds4_agent_test.c share one builder.
 *
 * The four tool lines are Pi's read, bash, edit and write (Pi, MIT; see
 * tests/gen_mellum_chat_ids.py).  Each is the bytes `tools | tojson` emits under transformers'
 * apply_chat_template (json.dumps with the default ", " and ": " separators, insertion order, no
 * key sorting, no HTML escaping).  tests/test_mellum_config.c compares what these functions build to the text jinja2
 * rendered (tests/gen_mellum_chat_ids.py), so a typo here is a red test, not a silent drift.
 *
 * The functions return the *contents* of a turn, not the <|im_start|>role / <|im_end|> framing:
 * ds4_chat_append_message adds that.  The caller frees the result. */
#ifndef MELLUM_TOOL_TEXT_H
#define MELLUM_TOOL_TEXT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MTT_N_TOOLS 4

static const char *const mtt_tool_json[MTT_N_TOOLS] = {
    "{\"type\": \"function\", \"function\": {\"name\": \"read\", \"description\": \"Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). Images are sent as attachments. For text files, output is truncated to 2000 lines or 50KB (whichever is hit first). Use offset/limit for large files. When you need the full file, continue with offset until complete.\""
    ", \"parameters\": {\"type\": \"object\", \"required\": [\"path\"], \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Path to the file to read (relative or absolute)\"}, \"offset\": {\"type\": \"number\", \"description\": \"Line number to start reading from (1-indexed)\"}, \"limit\": {\"type\": \"number\", \"description\": \"Maximum number of lines to read\"}}}, \"strict\": false}}",
    "{\"type\": \"function\", \"function\": {\"name\": \"bash\", \"description\": \"Execute a bash command in the current working directory. Returns stdout and stderr. Output is truncated to last 2000 lines or 50KB (whichever is hit first). If truncated, full output is saved to a temp file. Optionally provide a timeout in seconds.\""
    ", \"parameters\": {\"type\": \"object\", \"required\": [\"command\"], \"properties\": {\"command\": {\"type\": \"string\", \"description\": \"Shell command to execute\"}, \"timeout\": {\"type\": \"number\", \"description\": \"Timeout in seconds (optional, no default timeout)\"}}}, \"strict\": false}}",
    "{\"type\": \"function\", \"function\": {\"name\": \"edit\", \"description\": \"Edit a single file using exact text replacement. Every edits[].oldText must match a unique, non-overlapping region of the original file. If two changes affect the same block or nearby lines, merge them into one edit instead of emitting overlapping edits. Do not include large unchanged regions just to connect distant changes.\""
    ", \"parameters\": {\"type\": \"object\", \"required\": [\"path\", \"edits\"], \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Path to the file to edit (relative or absolute)\"}, \"edits\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"required\": [\"oldText\", \"newText\"], \"properties\": {\"oldText\": {\"type\": \"string\", \"description\": \"Exact text for one targeted replacement. It must be unique in the original file and must not overlap with any other edits[].oldText in the same call.\"}, \"newText\": {\"type\": \"string\", \"description\": \"Replacement text for this targeted edit.\"}}}, \"description\": \"One or more targeted replacements. Each edit is matched against the original file, not incrementally. Do not include overlapping or nested edits. If two changes touch the same block or nearby lines, merge them into one edit instead.\"}}}, \"strict\": false}}",
    "{\"type\": \"function\", \"function\": {\"name\": \"write\", \"description\": \"Write content to a file. Creates the file if it doesn't exist, overwrites if it does. Automatically creates parent directories.\""
    ", \"parameters\": {\"type\": \"object\", \"required\": [\"path\", \"content\"], \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Path to the file to write (relative or absolute)\"}, \"content\": {\"type\": \"string\", \"description\": \"Content to write to the file\"}}}, \"strict\": false}}",
};

static const char mtt_tools_head[] =
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
static const char mtt_tools_tail[] =
    "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

static char *mtt_dup_cat(const char *const *parts, int n) {
    size_t len = 1;
    for (int i = 0; i < n; i++) len += strlen(parts[i]);
    char *out = malloc(len), *p = out;
    if (!out) { fputs("mellum_tool_text: out of memory\n", stderr); exit(2); }
    for (int i = 0; i < n; i++) { size_t k = strlen(parts[i]); memcpy(p, parts[i], k); p += k; }
    *p = '\0';
    return out;
}

/* The system turn's content with tools: the template writes the system message (if there is one)
 * and a blank line, then the tools block.  A NULL `preamble` is "no system message"; an empty
 * string still writes the two newlines, as the template does for an empty system message. */
static char *mtt_system_text(const char *preamble) {
    const char *parts[2 + 2 * MTT_N_TOOLS + 2];
    int n = 0;
    if (preamble) { parts[n++] = preamble; parts[n++] = "\n\n"; }
    parts[n++] = mtt_tools_head;
    for (int i = 0; i < MTT_N_TOOLS; i++) { parts[n++] = "\n"; parts[n++] = mtt_tool_json[i]; }
    parts[n++] = mtt_tools_tail;
    return mtt_dup_cat(parts, n);
}

/* An assistant turn that carries reasoning and one tool call, empty content:
 *   <think>\n{reasoning}\n</think>\n\n<tool_call>\n{"name": "{name}", "arguments": {args}}\n</tool_call>
 * `reasoning` is stripped of leading and trailing newlines first (reasoning_content.strip('\n')); a
 * NULL or empty reasoning writes no think block, as the template does.  `args_json` is written
 * verbatim, which is what the template does for string arguments and what tojson gives a dict. */
static char *mtt_assistant_text(const char *reasoning, const char *name, const char *args_json) {
    const char *parts[9];
    int n = 0;
    char *r = NULL;
    if (reasoning && reasoning[0]) {
        const char *s = reasoning;
        size_t len = strlen(s);
        while (len && *s == '\n') { s++; len--; }
        while (len && s[len - 1] == '\n') len--;
        if (len) {
            r = malloc(len + 1);
            if (!r) { fputs("mellum_tool_text: out of memory\n", stderr); exit(2); }
            memcpy(r, s, len);
            r[len] = '\0';
            parts[n++] = "<think>\n"; parts[n++] = r; parts[n++] = "\n</think>\n\n";
        }
    }
    parts[n++] = "<tool_call>\n{\"name\": \"";
    parts[n++] = name;
    parts[n++] = "\", \"arguments\": ";
    parts[n++] = args_json;
    parts[n++] = "}\n</tool_call>";
    char *out = mtt_dup_cat(parts, n);
    free(r);
    return out;
}

#endif
