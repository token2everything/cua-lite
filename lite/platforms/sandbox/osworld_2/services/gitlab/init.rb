# Bootstrap through gitlab-rails runner; --health uses embedded Ruby without Rails.
# Never emit exceptions, request headers, response bodies, or credential values.
require 'json'
require 'net/http'
require 'uri'
require 'time'

started_at = Process.clock_gettime(Process::CLOCK_MONOTONIC)
diagnostic = {
  state: 'starting', stage: 'version_check', attempt: 0,
  sign_in_status: nil, api_status: nil, readiness_status: nil, error_class: nil,
  identity_verified: false
}
write_private = lambda do |name, contents|
  target = "/gitlab-output/#{name}"
  temporary = "#{target}.#{Process.pid}.tmp"
  begin
    File.open(temporary, File::WRONLY | File::CREAT | File::EXCL, 0o600) do |file|
      file.write(contents)
      file.flush
      file.fsync
    end
    File.chmod(0o600, temporary)
    raise 'Private file permissions not enforced' unless (File.stat(temporary).mode & 0o777) == 0o600
    File.rename(temporary, target)
  ensure
    File.unlink(temporary) if File.exist?(temporary)
  end
end

# Only numeric memory counters are retained, from this GitLab container's cgroup.
read_memory = lambda do
  memory = { scope: 'gitlab_container', version: nil, counters: {} }
  if File.readable?('/sys/fs/cgroup/memory.current')
    base = '/sys/fs/cgroup'
    memory[:version] = 2
    paths = { current_bytes: 'memory.current', limit_bytes: 'memory.max', peak_bytes: 'memory.peak' }
    counter_file = 'memory.events'
    allowed_counters = %w[low high max oom oom_kill oom_group_kill]
  elsif File.readable?('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    base = '/sys/fs/cgroup/memory'
    memory[:version] = 1
    paths = { current_bytes: 'memory.usage_in_bytes', limit_bytes: 'memory.limit_in_bytes',
              peak_bytes: 'memory.max_usage_in_bytes', limit_failures: 'memory.failcnt' }
    counter_file = 'memory.oom_control'
    allowed_counters = %w[oom_kill_disable under_oom oom_kill]
  else
    next memory
  end
  paths.each do |key, name|
    next unless File.readable?("#{base}/#{name}")
    value = File.read("#{base}/#{name}").strip
    memory[key] = value.match?(/\A[0-9]+\z/) ? value.to_i : (value == 'max' ? 'max' : nil)
  end
  if File.readable?("#{base}/#{counter_file}")
    File.foreach("#{base}/#{counter_file}") do |line|
      key, value = line.split
      memory[:counters][key] = value.to_i if allowed_counters.include?(key) && value&.match?(/\A[0-9]+\z/)
    end
  end
  memory
rescue StandardError => error
  { scope: 'gitlab_container', error_class: error.class.name }
end

# Both bootstrap and ongoing checks use loopback with the canonical Host header.
# Retain only status enums; dependency messages and response bodies stay private.
probe_health = lambda do |url, token_value, root_id|
  probe_started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  health = { ready: false, sign_in_status: nil, api_status: nil, readiness_status: nil,
             readiness_overall_status: nil,
             identity_verified: false, components: {}, component_count: 0,
             failed_component_count: 0, request_error_class: nil }
  begin
    canonical_uri = URI.parse(url)
    canonical_host = "#{canonical_uri.host}:#{canonical_uri.port}"
    http = Net::HTTP.new('127.0.0.1', 80, nil)
    http.open_timeout = 3
    http.read_timeout = 8
    http.max_retries = 0
    sign_in = http.get('/users/sign_in', 'Host' => canonical_host)
    health[:sign_in_status] = sign_in.code.to_i
    request = Net::HTTP::Get.new('/api/v4/user')
    request['Host'] = canonical_host
    request['PRIVATE-TOKEN'] = token_value
    response = http.request(request)
    health[:api_status] = response.code.to_i
    identity = response.code == '200' ? JSON.parse(response.body) : {}
    health[:identity_verified] = identity['id'] == root_id && identity['username'] == 'root' &&
      identity['is_admin'] == true

    response = http.get('/-/readiness?all=1', 'Host' => canonical_host)
    health[:readiness_status] = response.code.to_i
    if %w[200 503].include?(response.code)
      payload = JSON.parse(response.body)
      overall = payload.delete('status')
      health[:readiness_overall_status] = %w[ok failed].include?(overall) ? overall : 'unknown'
      health[:component_count] = payload.length
      # GitLab 18.7 Probes::Collection requires any successful probe per group.
      health[:failed_component_count] = payload.count do |_name, probes|
        !probes.is_a?(Array) || !probes.any? { |item| item.is_a?(Hash) && item['status'] == 'ok' }
      end
      allowed = %w[master_check db_check cache_check queues_check shared_state_check sessions_check
                   redis_check gitaly_check trace_chunks_check rate_limiting_check
                   repository_cache_check feature_flag_check workhorse_check load_balancing_check]
      payload.each do |name, values|
        next unless allowed.include?(name)
        health[:components][name] = (values.is_a?(Array) ? values : []).map do |item|
          status = item.is_a?(Hash) ? item['status'] : nil
          %w[ok failed].include?(status) ? status : 'unknown'
        end
      end
    end
    health[:ready] = (200..399).include?(health[:sign_in_status]) && health[:identity_verified] &&
      health[:readiness_status] == 200 && health[:readiness_overall_status] == 'ok' &&
      health[:component_count].positive? &&
      health[:failed_component_count].zero?
  rescue StandardError => error
    health[:request_error_class] = error.class.name
  end
  health[:probe_elapsed_seconds] = (Process.clock_gettime(Process::CLOCK_MONOTONIC) - probe_started).round(3)
  health
end

if ARGV == ['--health']
  health = { ready: false }
  begin
    connection = JSON.parse(File.read('/gitlab-output/connection.json'))
    initial = JSON.parse(File.read('/gitlab-output/readiness.json'))
    health = probe_health.call(connection.fetch('url'), connection.fetch('private_token'), initial.fetch('root_id'))
    health.merge!(url: connection.fetch('url'), version: initial.fetch('version'), auth_check_host: '127.0.0.1',
                  is_admin: health[:identity_verified])
  rescue StandardError => error
    health[:error_class] = error.class.name
  end
  begin
    health[:cgroup_memory] = read_memory.call
    health[:updated_at] = Time.now.utc.iso8601
    write_private.call('health.json', JSON.generate(health) + "\n")
  rescue StandardError => error
    warn "GitLab health diagnostic publication failed (#{error.class.name})"
    exit 1
  end
  exit(health[:ready] ? 0 : 2)
end

publish_diagnostic = lambda do
  diagnostic[:elapsed_seconds] = (Process.clock_gettime(Process::CLOCK_MONOTONIC) - started_at).round(3)
  diagnostic[:updated_at] = Time.now.utc.iso8601
  write_private.call('bootstrap.json', JSON.generate(diagnostic) + "\n")
end

begin
  publish_diagnostic.call
  raise 'Unexpected GitLab version' unless Gitlab::VERSION == '18.7.0'
  url = ENV.fetch('GITLAB_URL')
  token_value = ENV.fetch('GITLAB_PRIVATE_TOKEN')
  root_password = ENV.fetch('GITLAB_ROOT_PASSWORD')
  diagnostic[:stage] = 'root_account'
  publish_diagnostic.call
  root = User.find_by_username('root')
  raise 'Root account missing' unless root&.admin?
  root.password = root_password
  root.password_confirmation = root_password
  root.save!

  # Match the official Task-Web/gitlab root PAT provisioning contract.
  diagnostic[:stage] = 'token_provision'
  publish_diagnostic.call
  PersonalAccessToken.where(user: root, name: 'auto-generated-token').destroy_all
  root.personal_access_tokens.create!(
    name: 'auto-generated-token',
    token_digest: Gitlab::CryptoHelper.sha256(token_value),
    scopes: [:api, :read_user, :read_api, :read_repository, :write_repository, :sudo, :admin_mode],
    expires_at: 365.days.from_now
  )

  # Authenticate over loopback; nested-to-outer NAT is a separate network check.
  diagnostic[:stage] = 'loopback_auth'
  publish_diagnostic.call
  verified = false
  deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + 600
  while Process.clock_gettime(Process::CLOCK_MONOTONIC) < deadline
    diagnostic[:attempt] += 1
    diagnostic.merge!(probe_health.call(url, token_value, root.id))
    diagnostic[:cgroup_memory] = read_memory.call
    verified = diagnostic[:ready]
    publish_diagnostic.call
    break if verified
    sleep 5
  end
  raise 'Admin API readiness deadline exceeded' unless verified

  diagnostic[:stage] = 'publish_credentials'
  publish_diagnostic.call
  documents = {
    'connection.json' => {
      url: url,
      private_token: token_value,
      admin_username: 'root',
      admin_password: root_password
    },
    'readiness.json' => {
      ready: true,
      url: url,
      version: Gitlab::VERSION,
      root_id: root.id,
      auth_check_host: '127.0.0.1',
      sign_in_status: diagnostic[:sign_in_status],
      api_status: 200,
      readiness_status: diagnostic[:readiness_status],
      readiness_overall_status: diagnostic[:readiness_overall_status],
      components: diagnostic[:components],
      is_admin: true,
      verified_at: Time.now.utc.iso8601
    }
  }
  documents.each do |name, document|
    write_private.call(name, JSON.generate(document) + "\n")
  end
  shell_quote = ->(value) { "'" + value.gsub("'", "'\"'\"'") + "'" }
  write_private.call('connection.env',
    "GITLAB_URL=#{shell_quote.call(url)}\nGITLAB_PRIVATE_TOKEN=#{shell_quote.call(token_value)}\n")
  diagnostic[:state] = 'ready'
  diagnostic[:stage] = 'complete'
  publish_diagnostic.call
rescue StandardError => error
  # Class names are non-sensitive; exception messages may contain private values.
  diagnostic[:state] = 'failed'
  diagnostic[:error_class] = error.class.name
  begin
    publish_diagnostic.call
  rescue StandardError => diagnostic_error
    warn "GitLab diagnostic publication failed (#{diagnostic_error.class.name})"
  end
  warn "GitLab initialization failed (#{error.class.name})"
  exit 1
end
