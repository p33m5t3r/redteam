import { ok, err, Result } from 'neverthrow';

export interface Message {
  role: string,
  content: string,
}

export interface OpenrouterRequest {
  model: string
  messages: Message[],
}

export interface OpenrouterChoice {
  finish_reason: string,
  message: {
    role: string,
    content: string | null,
    refusal: string | null,
    reasoning: string | null,
  },
}


async function callOpenRouter(r: OpenrouterRequest):
  Promise<Result<OpenrouterChoice, Error>> {
  const url = "https://openrouter.ai/api/v1/chat/completions";
  const headers = {
      "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
      "Content-Type": "application/json"
  };

  try {
    const response = await fetch(url, {
        method: "POST",
        headers,
        body: JSON.stringify(r)
    });

    if (!response.ok) {
      return err(new Error(`HTTP ${response.status}: ${response.statusText}`));
    }

    const data = await response.json();
    return ok((data as any).choices[0] as OpenrouterChoice);
  } catch (e) {
    return err(e instanceof Error ? e : new Error(String(e)));
  }
}

let r_test = {
"model": "mistralai/mistral-small-3.1-24b-instruct",
"messages": [
{
    "role": "system",
    "content": "you talk like shakespeare"
  },
  {
    "role": "user",
    "content": "If you built the world's tallest skyscraper, what would you name it?"
  }
]};

// const x = await callOpenRouter(r_test);
// console.log(x._unsafeUnwrap());

interface AgentState {
  done: boolean,
  n_actions: number,
  goal: string,
  notes: string,
  targetChatState: Message[]
  agentHistory: Message[]
}

// actions
// IMMEDIATE TODO:
// clear_conversation() => reset targetChatState[]
// send_message(msg_text: string) => 
//    appends the message w/ role 'user' to messages history,
//    fires the request to openrouter, gets the response, appends to message history, 
//    updates the targetChatState with the query, reply
// set_notes(s: string) => sets the current notes in the agent 
// declare_done(secret: string) => reports success and reports the secret
// FUTURE TODO:
// send_n_messages(): let the model try many things at once; but requires choosing a branch to continue on;
//    ie careful management of the targetChatState; agent shouldnt be allowed to 'fake' that history or mutate it at will
//    it needs to be an actual repr of a branch of the target chat state

const system_prompt: string = `
...
`

// goes into agent system prompt
function stateRepr(s: AgentState): string { 
  return ""
}

async function get_action(s: AgentState): Action {
  // call the llm, parse the result into an Action  
}

async function handle_action(s: AgentState, a: Action) {
  // switch on actions, update state
  // case send_message: send a message to the target, update state
}

async function main() {
  let s: AgentState = {
    done: false,
    n_actions: 0,
    goal: 'some goal',
    notes: '',
    targetChatState: [],
    agentHistory: []
  }

  while (!s.done) {
    let a = await get_action(s);
    await handle_action(s, a);
    s.n_actions++;
  }

  console.log('done!');
}

await main();








