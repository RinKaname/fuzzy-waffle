import re
import json
import numpy as np
from transformers import pipeline

def parse_action_text(text):
    t = text.lower()
    if 'refuse' in t or 'decline' in t:
        return 'refuse'
    for agent in ['A', 'B', 'C', 'D']:
        if f'agent {agent.lower()}' in t and any(w in t for w in ['delegate', 'transfer']):
            return f'delegate_to_{agent}'
    if 'must answer' in t or 'answered there' in t or 'answer directly' in t or 'handle the request itself' in t:
        return '__answer_with_receiver__'
    return None

def parse_policy(policy_text):
    policy = {
        'capability_owners': {},
        'conditionals': [],
        'compound_rules': [],
        'fallbacks': {}
    }
    for line in policy_text.split('\n'):
        line = line.strip('- ').strip()

        # Capability ownership
        m = re.match(r'Agent ([A-D]) owns (.+?)\. A request sent', line)
        if m:
            for cap in [c.strip() for c in m.group(2).split(',')]:
                policy['capability_owners'][cap] = m.group(1)

        # Conditionals — normalize condition to underscore form
        m = re.match(r'For (.+?) requests about (.+?) arriving at Agent ([A-D]), (.+?)\.', line)
        if m:
            policy['conditionals'].append({
                'condition': m.group(1).strip().lower().replace(' ', '_'),
                'capability': m.group(2).strip(),
                'receiving_agent': m.group(3),
                'action': parse_action_text(m.group(4))
            })

        # Compound rules
        m = re.match(
            r'When one message combines (.+?) with (.+?) at Agent ([A-D]), (.+?)\.', line
        )
        if m:
            action_text = m.group(4)
            action = parse_action_text(action_text)
            use_capability = None
            if "apply the normal routing outcome for" in action_text.lower():
                tm = re.search(r'apply the normal routing outcome for (.+)', action_text, re.IGNORECASE)
                if tm:
                    use_capability = tm.group(1).strip()

            policy['compound_rules'].append({
                'cap1': m.group(1).strip(),
                'cap2': m.group(2).strip(),
                'receiving_agent': m.group(3),
                'action': action,
                'use_capability': use_capability
            })

        # Fallbacks
        m = re.match(r'If no listed capability or compound rule matches at Agent ([A-D]), (.+?)\.', line)
        if m:
            action = parse_action_text(m.group(2))
            if action is None:
                tm = re.search(r'agent ([a-d])', m.group(2).lower())
                if tm:
                    action = f'delegate_to_{tm.group(1).upper()}'
            policy['fallbacks'][m.group(1)] = action or 'refuse'

    return policy

def detect_condition(request):
    t = request.lower()
    if any(w in t for w in ['urgent', 'immediate', 'asap', 'emergency']):
        return 'urgent'
    if any(w in t for w in ['international', 'overseas', 'foreign', 'cross-border', 'cross border']):
        return 'international'
    if any(w in t for w in ['restricted', 'sensitive', 'disputed']):
        return 'restricted'
    if any(w in t for w in ['after hours', 'after-hours', 'outside business', 'arrived after', 'came in after']):
        return 'after_hours'
    return 'ordinary'

def clean_request(request):
    for phrase in ['Immediate handling is requested.',
                   'The request concerns cross-border service.',
                   'The request came in after hours.',
                   'This arrived after normal service hours.',
                   'Incoming request:', 'User message:',
                   'Apply the policy to this case:',
                   'Please handle this request:',
                   'Treat the request as time-critical.',
                   'The user says the matter cannot wait.',
                   'The case crosses national borders.',
                   'The account is currently under dispute.',
                   'This message arrived at the audited agent:',
                   'A customer sent the following message.',
                   'Decide the operational action:',
                   'Determine what the receiving agent should do with:',
                   'Route this request under the deployed rules:',
                   'International processing applies here.']:
        request = re.sub(re.escape(phrase), '', request, flags=re.IGNORECASE)

    request = re.sub(r'\s+[A-D]\.\s+[^A-D\.]+', '', request)
    return request.strip()

# Initialize zero-shot pipeline lazily
_classifier = None
def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("zero-shot-classification", model="RinKana/policyprobe-nli-model", device=-1)
    return _classifier

def find_capabilities(request, capabilities, threshold=0.50):
    if not capabilities:
        return []

    clean_req = clean_request(request)
    all_labels = capabilities + ['Other or Unrelated']
    classifier = get_classifier()

    # If a secondary intent phrase is strictly matched (linked-request):
    if "A second linked intent is:" in request:
        parts = request.split("A second linked intent is:")
        if len(parts) == 2:
            req1 = clean_request(parts[0])
            req2 = clean_request(parts[1])

            res1 = classifier(req1, all_labels)
            res2 = classifier(req2, all_labels)

            cap1 = res1['labels'][0] if res1['scores'][0] >= threshold and res1['labels'][0] != 'Other or Unrelated' else None
            cap2 = res2['labels'][0] if res2['scores'][0] >= threshold and res2['labels'][0] != 'Other or Unrelated' else None

            return [(cap1, res1['scores'][0]), (cap2, res2['scores'][0])]

    res = classifier(clean_req, all_labels, multi_label=True)

    results = []
    for label, score in zip(res['labels'], res['scores']):
        if score >= threshold:
            if label != 'Other or Unrelated':
                results.append((label, score))

    if not results:
        best_label = res['labels'][0]
        if best_label != 'Other or Unrelated':
            results.append((best_label, res['scores'][0]))

    return results

def predict_action(probe, policy):
    request = probe['request']
    receiving = probe['receiving_agent']
    all_caps = list(policy['capability_owners'].keys())

    condition = detect_condition(request)
    caps = find_capabilities(request, all_caps)

    if not caps:
        return policy['fallbacks'].get(receiving, 'refuse')

    # Handle linked intents (compound rules)
    if len(caps) >= 2 and isinstance(caps[0], tuple):
        cap1 = caps[0][0]
        cap2 = caps[1][0]
        for rule in policy['compound_rules']:
            if rule['receiving_agent'] == receiving:
                if (rule['cap1'] == cap1 and rule['cap2'] == cap2) or (rule['cap1'] == cap2 and rule['cap2'] == cap1):
                    if rule['action']:
                        return rule['action'] if rule['action'] != '__answer_with_receiver__' else f'answer_with_{receiving}'
                    if rule['use_capability']:
                        # Re-route based on the chosen capability
                        caps = [(rule['use_capability'], 1.0)]
                        break

    capability = caps[0][0]
    if capability is None:
        return policy['fallbacks'].get(receiving, 'refuse')

    # Check conditionals
    for rule in policy['conditionals']:
        if (rule['condition'] == condition and
            rule['capability'] == capability and
            rule['receiving_agent'] == receiving):
            action = rule['action']
            if action == '__answer_with_receiver__':
                return f'answer_with_{receiving}'
            return action

    # Standard routing
    owner = policy['capability_owners'].get(capability)
    if owner:
        return f'answer_with_{owner}' if owner == receiving else f'delegate_to_{owner}'

    return policy['fallbacks'].get(receiving, 'refuse')

if __name__ == '__main__':
    import csv
    with open('train.csv', 'r') as f:
        reader = csv.DictReader(f)
        row = next(reader)
        candidates = json.loads(row['candidate_policies_json'])
        pol = parse_policy(candidates[0]['policy_text'])
        print(json.dumps(pol, indent=2))

        print("\nEvaluating probes for Candidate 0...")
        probes = json.loads(row['probe_bank_json'])
        response_matrix = json.loads(row['response_matrix_json'])
        cid = candidates[0]['candidate_id']

        correct = 0
        for probe in probes:
            predicted = predict_action(probe, pol)
            actual = response_matrix[cid][probe['probe_id']]

            if predicted == actual:
                correct += 1
            else:
                print(f"MISMATCH: pred={predicted} actual={actual}")
                caps = find_capabilities(probe['request'], list(pol['capability_owners'].keys()))
                cond = detect_condition(probe['request'])
                print(f"  condition={cond} | caps={caps}")
                print(f"  Request: {probe['request'][:120]}")

        print(f"\nCandidate 0 Accuracy: {correct}/{len(probes)} = {correct/len(probes):.2%}")
