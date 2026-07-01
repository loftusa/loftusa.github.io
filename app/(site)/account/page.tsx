import { auth, signIn, signOut } from "@/auth";

export const metadata = { title: "Account" };
export const dynamic = "force-dynamic";

export default async function AccountPage() {
  const session = await auth();

  return (
    <main style={{ maxWidth: "42rem", margin: "3rem auto", padding: "0 1.5rem" }}>
      <h1>Account</h1>
      {session?.user ? (
        <>
          <p>
            Signed in as <strong>{session.user.email}</strong>.
          </p>
          <form
            action={async () => {
              "use server";
              await signOut({ redirectTo: "/account/" });
            }}
          >
            <button type="submit">Sign out</button>
          </form>
        </>
      ) : (
        <>
          <p>Sign in with GitHub to save preferences (personalization coming soon).</p>
          <form
            action={async () => {
              "use server";
              await signIn("github", { redirectTo: "/account/" });
            }}
          >
            <button type="submit">Sign in with GitHub</button>
          </form>
        </>
      )}
    </main>
  );
}
